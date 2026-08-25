from __future__ import annotations

from collections import deque
from time import perf_counter


class InferenceMonitor:
    def __init__(self, latency_window: int = 200) -> None:
        self.started_at = perf_counter()
        self.latencies_ms: deque[float] = deque(maxlen=latency_window)
        self.total_requests = 0
        self.total_sign_requests = 0
        self.total_stt_requests = 0
        self.total_errors = 0

    def record_sign(self, latency_ms: float) -> None:
        self.total_requests += 1
        self.total_sign_requests += 1
        self.latencies_ms.append(latency_ms)

    def record_stt(self, latency_ms: float) -> None:
        self.total_requests += 1
        self.total_stt_requests += 1
        self.latencies_ms.append(latency_ms)

    def record_error(self) -> None:
        self.total_errors += 1

    def snapshot(self) -> dict[str, float | int]:
        avg_latency = (
            sum(self.latencies_ms) / len(self.latencies_ms)
            if self.latencies_ms
            else 0.0
        )
        uptime_s = perf_counter() - self.started_at
        return {
            "uptime_seconds": round(uptime_s, 2),
            "total_requests": self.total_requests,
            "sign_requests": self.total_sign_requests,
            "stt_requests": self.total_stt_requests,
            "total_errors": self.total_errors,
            "avg_latency_ms": round(avg_latency, 2),
            "latency_samples": len(self.latencies_ms),
        }
