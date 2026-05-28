import asyncio
import io
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import contextlib

import numpy as np
import soundfile as sf
import torch

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.audio import numpy_to_audiosegment
DEFAULT_NUM_STEP  = 32
DEFAULT_GUIDANCE_SCALE = 2.0
logger = logging.getLogger("tts.serve")


@dataclass
class InferenceRequest:
    text: str
    language: Optional[str] = None
    instruct: Optional[str] = None
    ref_audio_bytes: Optional[bytes] = None
    ref_text: Optional[str] = None
    speed: Optional[float] = None
    duration: Optional[float] = None
    max_duration_ms: Optional[float] = None
    request_timeout_s: Optional[float] = None
    num_step: int = 0
    guidance_scale: float = 0
    future: Optional[asyncio.Future] = field(default=None, repr=False)
    submit_time: float = field(default_factory=time.monotonic)


class BatchInferenceEngine:
    """Collects requests and processes them in batches for GPU efficiency."""

    def __init__(
        self,
        model: OmniVoice,
        max_batch_size: int = 8,
        max_wait_ms: float = 200,
        max_queue_size: int = 64,
    ):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_s = max_wait_ms / 1000.0
        self.max_queue_size = max_queue_size
        self._queue: asyncio.Queue[InferenceRequest] = asyncio.Queue(
            maxsize=max_queue_size if max_queue_size > 0 else 0
        )
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._worker_task: Optional[asyncio.Task] = None

    def start(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._worker_task = loop.create_task(self._batch_worker())

    def stop(self):
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    async def submit(self, req: InferenceRequest) -> bytes:
        if self._queue.full():
            raise RuntimeError(
                f"Server overloaded: {self.max_queue_size} requests queued. "
                "Please retry later."
            )
        if not req.duration or req.duration <= 0:
            req.duration = None
        if not req.max_duration_ms or req.max_duration_ms <= 0:
            req.max_duration_ms = None
        if not req.request_timeout_s or req.request_timeout_s <= 0:
            req.request_timeout_s = None
        if not req.num_step or req.num_step <= 0:
            req.num_step = DEFAULT_NUM_STEP
        if not req.guidance_scale or req.guidance_scale <= 0:
            req.guidance_scale = DEFAULT_GUIDANCE_SCALE
        if not req.speed or req.speed <= 0:
            req.speed = None
        if not req.language:
            req.language = None
        if not req.instruct:
            req.instruct = None
        if not req.ref_text:
            req.ref_text = None
        if not req.ref_audio_bytes:
            req.ref_audio_bytes = None

        future = self._loop.create_future()
        req.future = future
        await self._queue.put(req)

        try:
            if req.request_timeout_s is None:
                return await future
            return await asyncio.wait_for(future, timeout=req.request_timeout_s)
        except asyncio.TimeoutError:
            future.cancel()
            raise RuntimeError(
                f"Request timed out after {req.request_timeout_s}s. "
                f"Queue depth was {self._queue.qsize()}."
            )

    async def _batch_worker(self):
        """Continuously drains the queue and runs batched inference."""
        while True:
            batch: list[InferenceRequest] = []

            first = await self._queue.get()
            batch.append(first)

            deadline = time.monotonic() + self.max_wait_s
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    req = await asyncio.wait_for(
                        self._queue.get(), timeout=remaining
                    )
                    batch.append(req)
                except asyncio.TimeoutError:
                    break

            await self._process_batch(batch)

    async def _process_batch(self, batch: list[InferenceRequest]):
        try:
            results = await asyncio.get_event_loop().run_in_executor(
                None, self._sync_generate, batch
            )
            for req, wav_bytes in zip(batch, results):
                if not req.future.done():
                    req.future.set_result(wav_bytes)
        except Exception as e:
            logger.exception("Batch inference failed")
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)

    def _sync_generate(self, batch: list[InferenceRequest]) -> list[bytes]:
        voice_prompts = []
        gen_config = OmniVoiceGenerationConfig(
            num_step=batch[0].num_step,
            guidance_scale=batch[0].guidance_scale,
        )

        for req in batch:
            prompt = None
            with contextlib.suppress(Exception):
                if req.ref_audio_bytes is not None:
                    waveform, sr = self._decode_ref_audio(req.ref_audio_bytes)
                    prompt = self.model.create_voice_clone_prompt(
                        ref_audio=(waveform, sr),
                        ref_text=req.ref_text,
                    )

            voice_prompts.append(prompt)
        
        has_clone = any(p is not None for p in voice_prompts)
        has_not_clone = any(p is None for p in voice_prompts)

        if has_clone and has_not_clone:
            return self._generate_individually(batch, gen_config)

        kwargs = {}
        if has_clone:
            kwargs["voice_clone_prompt"] = voice_prompts
        if any(r.instruct for r in batch):
            kwargs["instruct"] = [r.instruct for r in batch]
        if any(r.speed for r in batch):
            kwargs["speed"] = [r.speed for r in batch]
        if any(r.max_duration_ms for r in batch):
            kwargs["max_duration_ms"] = [r.max_duration_ms for r in batch]
        if any(r.duration for r in batch):
            kwargs["duration"] = [r.duration for r in batch]
        if any(r.language for r in batch):
            kwargs["language"] = [r.language for r in batch]

        kwargs["generation_config"] = gen_config
        kwargs["text"] = [r.text for r in batch]

        with torch.inference_mode():
            wavs = self.model.generate(**kwargs)

        return [self._tensor_to_wav_bytes(w) for w in wavs]

    def _generate_individually(
        self, batch: list[InferenceRequest], gen_config
    ) -> list[bytes]:
        results = []
        for req in batch:
            kwargs = {}
            with contextlib.suppress(Exception):
                if req.ref_audio_bytes:
                    waveform, sr = self._decode_ref_audio(req.ref_audio_bytes)
                    kwargs["voice_clone_prompt"] = self.model.create_voice_clone_prompt(
                        ref_audio=(waveform, sr), ref_text=req.ref_text,
                    )
       
            if req.instruct:
                kwargs["instruct"] = req.instruct
            if req.speed:
                kwargs["speed"] = req.speed
            if req.max_duration_ms:
                kwargs["max_duration_ms"] = req.max_duration_ms
            if req.duration:
                kwargs["duration"] = req.duration
            if req.generation_config:
                kwargs["generation_config"] = gen_config
            if req.text:
                kwargs["text"] = req.text
            if req.language:
                kwargs["language"] = req.language

            with torch.inference_mode():
                wavs = self.model.generate(**kwargs)
            results.append(self._tensor_to_wav_bytes(wavs[0]))
        return results

    def _decode_ref_audio(self, audio_bytes: bytes) -> tuple[torch.Tensor, int]:
        try:
            wav_data, sr = sf.read(io.BytesIO(audio_bytes))
        except Exception as e:
            raise RuntimeError(
                f"invalid ref_audio: unsupported or corrupted audio format: {type(audio_bytes)}"
            ) from e

        if wav_data.ndim > 1:
            wav_data = wav_data.mean(axis=1)
        waveform = torch.from_numpy(wav_data).float().unsqueeze(0)
        return waveform, sr

    def _tensor_to_wav_bytes(self, wav: torch.Tensor | np.ndarray) -> bytes:
        if isinstance(wav, torch.Tensor):
            arr = wav.detach().cpu().numpy()
        else:
            arr = np.asarray(wav)

        arr = np.squeeze(arr)
        if arr.ndim > 1:
            # (channels, time) or (time, channels) -> mono (time,)
            arr = arr.mean(axis=0 if arr.shape[0] < arr.shape[-1] else -1)

        buf = io.BytesIO()
        segment = numpy_to_audiosegment(
            arr.astype(np.float32)[np.newaxis, :],
            self.model.sampling_rate,
        )
        segment.export(buf, format="mp3", bitrate="64k")
        return buf.getvalue()