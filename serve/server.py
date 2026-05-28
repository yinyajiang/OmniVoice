"""gRPC worker server — one instance per GPU."""

import asyncio
import io
import logging
import time

import grpc
import soundfile as sf
import torch
from pydub import AudioSegment

from omnivoice import OmniVoice

from . import omnivoice_pb2 as pb2
from . import omnivoice_pb2_grpc as pb2_grpc
from .config import (
    CFG_COMPILE_MODEL,
    CFG_DEVICE,
    CFG_DTYPE,
    CFG_LOAD_ASR_MODEL,
    CFG_MAX_BATCH,
    CFG_MAX_PENDING_QUEUE,
    CFG_MAX_BATCH_WAIT_MS,
    CFG_MODEL_PATH,
    CFG_GRPC_MAX_MESSAGE_LENGTH
)
from .engine import BatchInferenceEngine, InferenceRequest

logger = logging.getLogger("tts.serve")


def _str2bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


class OmniVoiceTTSServicer(pb2_grpc.OmniVoiceTTSServicer):

    def __init__(self, engine: BatchInferenceEngine):
        self._engine = engine

    async def Synthesize(self, request, context):
        text = request.text.strip()
        if not text:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "text is required")

        req = InferenceRequest(
            text=text,
            language=request.language if request.HasField("language") else None,
            instruct=request.instruct if request.HasField("instruct") else None,
            speed=request.speed if request.HasField("speed") else None,
            duration=request.duration if request.HasField("duration") else None,
            max_duration_ms=(
                request.max_duration_ms
                if request.HasField("max_duration_ms")
                else None
            ),
            request_timeout_s=(
                request.request_timeout_s
                if request.HasField("request_timeout_s")
                else None
            ),
            num_step=max(4, min(64, request.num_step)) if request.num_step else 32,
            guidance_scale=request.guidance_scale if request.guidance_scale else 2.0,
            ref_audio_bytes=request.ref_audio if request.HasField("ref_audio") else None,
            ref_text=request.ref_text if request.HasField("ref_text") else None,
        )

        t0 = time.monotonic()
        try:
            wav_bytes = await self._engine.submit(req)
        except RuntimeError as e:
            msg = str(e)
            if "overloaded" in msg:
                await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, msg)
            elif "timed out" in msg:
                await context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, msg)
            elif "invalid ref_audio" in msg:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, msg)
            else:
                await context.abort(grpc.StatusCode.INTERNAL, msg)
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

        return pb2.SynthesizeResponse(
            audio=wav_bytes,
            elapsed_s=time.monotonic() - t0,
        )

    async def SynthesizeTest(self, request, context):
        synth_resp = await self.Synthesize(request, context)
        audio = AudioSegment.from_file(io.BytesIO(synth_resp.audio), format="mp3")
        audio_duration_s = len(audio) / 1000.0
        return pb2.SynthesizeTestResponse(
            audio_duration_s=audio_duration_s,
            elapsed_s=synth_resp.elapsed_s,
        )

    async def Asr(self, request, context):
        if not request.audio:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "audio is required"
            )

        try:
            wav_data, sr = sf.read(io.BytesIO(request.audio))
        except Exception:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "invalid audio: unsupported or corrupted format",
            )

        if wav_data.ndim > 1:
            wav_data = wav_data.mean(axis=1)
        waveform = torch.from_numpy(wav_data).float().unsqueeze(0)

        t0 = time.monotonic()
        try:
            text = await asyncio.get_event_loop().run_in_executor(
                None, self._engine.model.transcribe, (waveform, sr)
            )
        except RuntimeError as e:
            msg = str(e)
            if "ASR model is not loaded" in msg:
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, msg)
            await context.abort(grpc.StatusCode.INTERNAL, msg)
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

        return pb2.AsrResponse(
            text=text,
            elapsed_s=time.monotonic() - t0,
        )

    async def Health(self, request, context):
        return pb2.HealthResponseArray(
            items=[
                pb2.HealthResponse(
                    status="ok",
                    device=CFG_DEVICE,
                    model=CFG_MODEL_PATH,
                    queue_depth=self._engine.pending_count,
                    max_queue_size=CFG_MAX_PENDING_QUEUE,
                )
            ],
        )


async def serve(port: int, load_asr: bool = True):
    logger.info(
        "Loading model=%s device=%s dtype=%s compile=%s",
        CFG_MODEL_PATH, CFG_DEVICE, CFG_DTYPE, CFG_COMPILE_MODEL,
    )

    model = OmniVoice.from_pretrained(
        CFG_MODEL_PATH,
        device_map=CFG_DEVICE,
        dtype=CFG_DTYPE,
        load_asr=load_asr,
        asr_model_name=CFG_LOAD_ASR_MODEL,
    )

    if CFG_COMPILE_MODEL and CFG_DEVICE.startswith("cuda"):
        logger.info("Compiling LLM backbone with torch.compile ...")
        model.llm = torch.compile(model.llm, mode="reduce-overhead")
        with torch.inference_mode():
            model.generate(text="warmup", language="en")
        logger.info("Compilation and warmup done.")

    engine = BatchInferenceEngine(
        model,
        max_batch_size=CFG_MAX_BATCH,
        max_wait_ms=CFG_MAX_BATCH_WAIT_MS,
        max_queue_size=CFG_MAX_PENDING_QUEUE,
    )
    engine.start(asyncio.get_event_loop())

    server = grpc.aio.server(options=[
        ("grpc.max_send_message_length", CFG_GRPC_MAX_MESSAGE_LENGTH),
        ("grpc.max_receive_message_length", CFG_GRPC_MAX_MESSAGE_LENGTH),
    ])
    pb2_grpc.add_OmniVoiceTTSServicer_to_server(
        OmniVoiceTTSServicer(engine), server
    )
    server.add_insecure_port(f"127.0.0.1:{port}")
    await server.start()

    logger.info(
        "gRPC worker ready on port %d\n"
        "  model=%s  device=%s  dtype=%s\n"
        "  max_batch=%d  max_wait_ms=%.0f  max_queue=%d\n"
        "  grpc_max_message_length=%d\n"
        "  compile=%s  load_asr=%s  load_asr_model=%s",
        port,
        CFG_MODEL_PATH, CFG_DEVICE, CFG_DTYPE,
        CFG_MAX_BATCH, CFG_MAX_BATCH_WAIT_MS, CFG_MAX_PENDING_QUEUE,
        CFG_GRPC_MAX_MESSAGE_LENGTH,
        CFG_COMPILE_MODEL, load_asr,
        CFG_LOAD_ASR_MODEL,
    )

    try:
        await server.wait_for_termination()
    finally:
        engine.stop()
        logger.info("Worker shut down.")


def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="OmniVoice gRPC worker")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--load-asr", type=_str2bool, default=False)
    args = parser.parse_args()

    asyncio.run(serve(args.port, args.load_asr))


if __name__ == "__main__":
    main()
