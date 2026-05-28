"""gRPC gateway — round-robin proxy in front of multiple workers."""

import asyncio
import logging

import grpc

from . import omnivoice_pb2 as pb2
from . import omnivoice_pb2_grpc as pb2_grpc
from .config import CFG_GRPC_MAX_MESSAGE_LENGTH

logger = logging.getLogger("omnivoice.gateway")

_GRPC_OPTIONS = [
    ("grpc.max_send_message_length", CFG_GRPC_MAX_MESSAGE_LENGTH),
    ("grpc.max_receive_message_length", CFG_GRPC_MAX_MESSAGE_LENGTH),
]


class GatewayServicer(pb2_grpc.OmniVoiceTTSServicer):

    def __init__(self, worker_addrs: list[str]):
        if not worker_addrs:
            raise ValueError("worker_addrs is required")
        self._channels: list[grpc.aio.Channel] = []
        self._stubs: list[pb2_grpc.OmniVoiceTTSStub] = []
        self._asr_enabled: list[bool] = []
        for addr in worker_addrs:
            ch_addr, asr_enabled = self._parse_worker_addr(addr)
            ch = grpc.aio.insecure_channel(ch_addr, options=_GRPC_OPTIONS)
            self._channels.append(ch)
            self._stubs.append(pb2_grpc.OmniVoiceTTSStub(ch))
            self._asr_enabled.append(asr_enabled)
        self._in_flight = [0] * len(self._stubs)
        self._pick_lock = asyncio.Lock()

    @staticmethod
    def _parse_worker_addr(raw: str) -> tuple[str, bool]:
        s = raw.strip()
        if s.endswith("-asr"):
            return s[: -len("-asr")], True
        return s, False

    async def _acquire_stub(self, *, asr_only: bool = True) -> tuple[int, pb2_grpc.OmniVoiceTTSStub]:
        async with self._pick_lock:
            # Prefer the first idle worker; if none idle, choose least busy.
            candidates = (
                [i for i, ok in enumerate(self._asr_enabled) if ok]
                if asr_only
                else list(range(len(self._in_flight)))
            )
            if not candidates:
                raise RuntimeError("No ASR-enabled workers")

            for i in candidates:
                n = self._in_flight[i]
                if n == 0:
                    self._in_flight[i] += 1
                    return i, self._stubs[i]

            i = min(candidates, key=self._in_flight.__getitem__)
            self._in_flight[i] += 1
            return i, self._stubs[i]

    async def _release_stub(self, idx: int):
        async with self._pick_lock:
            self._in_flight[idx] -= 1

    async def Synthesize(self, request, context):
        idx, stub = await self._acquire_stub(asr_only=not bool(request.ref_text))
        try:
            resp = await stub.Synthesize(request)
            resp.worker_id = f"worker-{idx}"
            return resp
        except grpc.aio.AioRpcError as e:
            await context.abort(e.code(), e.details())
        finally:
            await self._release_stub(idx)

    async def SynthesizeTest(self, request, context):
        idx, stub = await self._acquire_stub(asr_only=not bool(request.ref_text))
        try:
            resp = await stub.SynthesizeTest(request)
            resp.worker_id = f"worker-{idx}"
            return resp
        except grpc.aio.AioRpcError as e:
            await context.abort(e.code(), e.details())
        finally:
            await self._release_stub(idx)

    async def Asr(self, request, context):
        try:
            idx, stub = await self._acquire_stub(asr_only=True)
        except RuntimeError as e:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(e))
        try:
            resp = await stub.Asr(request)
            resp.worker_id = f"worker-{idx}"
            return resp
        except grpc.aio.AioRpcError as e:
            await context.abort(e.code(), e.details())
        finally:
            await self._release_stub(idx)

    async def Health(self, request, context):
        items = []
        for i, stub in enumerate(self._stubs):
            resp = await stub.Health(pb2.HealthRequest())
            items.extend(resp.items)
        if not items:
            await context.abort(
                grpc.StatusCode.UNAVAILABLE, "No healthy workers"
            )

        return pb2.HealthResponseArray(
            workers=len(self._stubs),
            load_asr_workers=sum(1 for ok in self._asr_enabled if ok),
            items=items,
        )

    async def close(self):
        for ch in self._channels:
            await ch.close()


async def serve_gateway(listen_port: int, worker_addrs: list[str]):
    servicer = GatewayServicer(worker_addrs)

    server = grpc.aio.server(options=_GRPC_OPTIONS)
    pb2_grpc.add_OmniVoiceTTSServicer_to_server(servicer, server)
    server.add_insecure_port(f"0.0.0.0:{listen_port}")
    await server.start()

    try:
        await server.wait_for_termination()
    except asyncio.CancelledError:
        # Normal path when Ctrl+C cancels asyncio.run().
        pass
    finally:
        # Explicitly stop gRPC server before event loop closes to avoid
        # "AioServer.shutdown was never awaited" warnings.
        await server.stop(grace=1.0)
        await servicer.close()
        logger.info("Gateway shut down.")


def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="OmniVoice gRPC gateway")
    parser.add_argument("--port", type=int, default=50050,
                        help="Port the gateway listens on")
    parser.add_argument("--workers", nargs="+", required=True,
                        help="Worker addresses, e.g. 127.0.0.1:50051-asr 127.0.0.1:50052-asr")
    args = parser.parse_args()

    asyncio.run(serve_gateway(args.port, args.workers))


if __name__ == "__main__":
    main()
