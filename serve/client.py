"""Command-line gRPC client for OmniVoice TTS."""

import argparse
import asyncio
import logging
import os
import time

import grpc

from . import omnivoice_pb2 as pb2
from . import omnivoice_pb2_grpc as pb2_grpc

logger = logging.getLogger("omnivoice.client")


async def synthesize(
    stub: pb2_grpc.OmniVoiceTTSStub,
    text: str,
    language: str | None = None,
    instruct: str | None = None,
    speed: float | None = None,
    max_duration_ms: float | None = None,
    request_timeout_s: float | None = None,
    num_step: int = 32,
    guidance_scale: float = 2.0,
    ref_audio_path: str | None = None,
    ref_text: str | None = None,
    timeout: float = 300,
) -> tuple[bytes, float, str]:
    kwargs: dict = {
        "text": text,
        "num_step": num_step,
        "guidance_scale": guidance_scale,
    }
    if language:
        kwargs["language"] = language
    if instruct:
        kwargs["instruct"] = instruct
    if speed is not None:
        kwargs["speed"] = speed
    if max_duration_ms is not None:
        kwargs["max_duration_ms"] = max_duration_ms
    if request_timeout_s is not None:
        kwargs["request_timeout_s"] = request_timeout_s
    if ref_text:
        kwargs["ref_text"] = ref_text
    if ref_audio_path:
        with open(ref_audio_path, "rb") as f:
            kwargs["ref_audio"] = f.read()

    request = pb2.SynthesizeRequest(**kwargs)
    response = await stub.Synthesize(request, timeout=timeout)
    return response.audio, response.elapsed_s, response.worker_id


async def run_single(args):
    async with grpc.aio.insecure_channel(args.server) as channel:
        stub = pb2_grpc.OmniVoiceTTSStub(channel)

        t0 = time.monotonic()
        audio, elapsed_s, worker_id = await synthesize(
            stub,
            text=args.text,
            language=args.language,
            instruct=args.instruct,
            speed=args.speed,
            max_duration_ms=args.max_duration_ms,
            request_timeout_s=args.request_timeout_s,
            num_step=args.num_step,
            guidance_scale=args.guidance_scale,
            ref_audio_path=args.ref_audio,
            ref_text=args.ref_text,
        )
        elapsed = time.monotonic() - t0

        with open(args.output, "wb") as f:
            f.write(audio)

        print(
            f"Done: {args.output}  "
            f"size={len(audio)} bytes  "
            f"elapsed={elapsed:.1f}s(client)  "
            f"server_elapsed={elapsed_s:.2f}s  "
            f"worker_id={worker_id}"
        )


async def run_concurrent(args):
    async with grpc.aio.insecure_channel(args.server) as channel:
        stub = pb2_grpc.OmniVoiceTTSStub(channel)

        async def one_request(idx: int):
            t0 = time.monotonic()
            try:
                audio, _, worker_id = await synthesize(
                    stub,
                    text=args.text,
                    language=args.language,
                    instruct=args.instruct,
                    speed=args.speed,
                    max_duration_ms=args.max_duration_ms,
                    request_timeout_s=args.request_timeout_s,
                    num_step=args.num_step,
                    guidance_scale=args.guidance_scale,
                    ref_audio_path=args.ref_audio,
                    ref_text=args.ref_text,
                )
                elapsed = time.monotonic() - t0
                out = f"output_{idx}.wav"
                with open(out, "wb") as f:
                    f.write(audio)
                return idx, "OK", worker_id, elapsed, len(audio)
            except grpc.aio.AioRpcError as e:
                elapsed = time.monotonic() - t0
                return idx, f"ERR:{e.code().name}", "-", elapsed, 0

        tasks = [one_request(i) for i in range(1, args.concurrency + 1)]
        results = await asyncio.gather(*tasks)

    print("=" * 72)
    print(f"{'#':<5} {'Status':<18} {'Worker':<16} {'Time':>8} {'Size':>10}")
    print("-" * 72)

    total_t = 0.0
    ok = 0
    for idx, status, worker_id, elapsed, size in sorted(results):
        print(
            f"{idx:<5} {status:<18} {worker_id:<16} "
            f"{elapsed:>7.1f}s {size:>9} B"
        )
        total_t += elapsed
        if status.startswith("OK"):
            ok += 1

    n = len(results)
    print("-" * 72)
    print(f"Success: {ok}/{n}  Avg: {total_t / n:.1f}s")
    print("=" * 72)


async def run_health(args):
    async with grpc.aio.insecure_channel(args.server) as channel:
        stub = pb2_grpc.OmniVoiceTTSStub(channel)
        resp = await stub.Health(pb2.HealthRequest(), timeout=5)
        print(f"workers={resp.workers}  healthy={len(resp.items)}")
        for idx, item in enumerate(resp.items):
            print(
                f"[{idx}] status={item.status}  device={item.device}  "
                f"model={item.model}  queue={item.queue_depth}/{item.max_queue_size}"
            )


async def run_asr(args):
    if not args.audio:
        raise ValueError("--audio is required")

    with open(args.audio, "rb") as f:
        audio_bytes = f.read()

    async with grpc.aio.insecure_channel(args.server) as channel:
        stub = pb2_grpc.OmniVoiceTTSStub(channel)
        t0 = time.monotonic()
        resp = await stub.Asr(pb2.AsrRequest(audio=audio_bytes), timeout=60)
        elapsed = time.monotonic() - t0
        print(
            f"text={resp.text}\n"
            f"elapsed={elapsed:.2f}s(client)  "
            f"server_elapsed={resp.elapsed_s:.2f}s  "
            f"worker_id={resp.worker_id}"
        )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="OmniVoice gRPC client")
    parser.add_argument("--server", default="localhost:50050",
                        help="Gateway address (default: localhost:50050)")
    sub = parser.add_subparsers(dest="cmd")

    # ── tts ──
    p_tts = sub.add_parser("tts", help="Single TTS request")
    p_tts.add_argument("--text", required=True)
    p_tts.add_argument("--language", default=None)
    p_tts.add_argument("--instruct", default=None)
    p_tts.add_argument("--speed", type=float, default=None)
    p_tts.add_argument("--max-duration-ms", type=float, default=None)
    p_tts.add_argument("--request-timeout-s", type=float, default=None)
    p_tts.add_argument("--num-step", type=int, default=32)
    p_tts.add_argument("--guidance-scale", type=float, default=2.0)
    p_tts.add_argument("--ref-audio", default=None)
    p_tts.add_argument("--ref-text", default=None)
    p_tts.add_argument("--output", "-o", default="output.wav")

    # ── bench ──
    p_bench = sub.add_parser("bench", help="Concurrent TTS benchmark")
    p_bench.add_argument("--text", required=True)
    p_bench.add_argument("--language", default=None)
    p_bench.add_argument("--instruct", default=None)
    p_bench.add_argument("--speed", type=float, default=None)
    p_bench.add_argument("--max-duration-ms", type=float, default=None)
    p_bench.add_argument("--request-timeout-s", type=float, default=None)
    p_bench.add_argument("--num-step", type=int, default=32)
    p_bench.add_argument("--guidance-scale", type=float, default=2.0)
    p_bench.add_argument("--ref-audio", default=None)
    p_bench.add_argument("--ref-text", default=None)
    p_bench.add_argument("--concurrency", "-n", type=int, default=5)

    # ── health ──
    sub.add_parser("health", help="Check server health")

    # ── asr ──
    p_asr = sub.add_parser("asr", help="ASR from audio file")
    p_asr.add_argument("--audio", required=True)

    args = parser.parse_args()

    if args.cmd == "tts":
        asyncio.run(run_single(args))
    elif args.cmd == "bench":
        asyncio.run(run_concurrent(args))
    elif args.cmd == "health":
        asyncio.run(run_health(args))
    elif args.cmd == "asr":
        asyncio.run(run_asr(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
