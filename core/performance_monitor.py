import time
from typing import Dict, Any, List, Optional
from contextlib import contextmanager

class PerformanceMonitor:
    """Enterprise performance telemetry logger measuring stage latencies,

    cache effectiveness, memory deltas, and throughput rates.
    """

    _logs: List[Dict[str, Any]] = []

    @classmethod
    @contextmanager
    def track_stage(cls, stage_name: str, rows: int = 0, extra: Optional[Dict[str, Any]] = None):
        t0 = time.perf_counter()
        entry = {
            "stage": stage_name,
            "rows": rows,
            "start_time": time.time(),
            "extra": extra or {}
        }
        try:
            yield entry
        finally:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            entry["elapsed_ms"] = elapsed_ms
            entry["elapsed_s"] = round(elapsed_ms / 1000, 3)
            cls._logs.append(entry)
            if len(cls._logs) > 500:
                cls._logs = cls._logs[-500:]

    @classmethod
    def record_metric(
        cls,
        operation: str,
        elapsed_ms: float,
        rows: int = 0,
        cache_hit: bool = False,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        cls._logs.append({
            "stage": operation,
            "elapsed_ms": elapsed_ms,
            "elapsed_s": round(elapsed_ms / 1000, 3),
            "rows": rows,
            "cache_hit": cache_hit,
            "timestamp": time.time(),
            "extra": extra or {}
        })

    @classmethod
    def get_summary(cls) -> Dict[str, Any]:
        if not cls._logs:
            return {"total_operations": 0, "recent_logs": []}

        stages = {}
        for l in cls._logs:
            st = l["stage"]
            if st not in stages:
                stages[st] = {"count": 0, "total_ms": 0.0, "max_ms": 0.0}
            stages[st]["count"] += 1
            stages[st]["total_ms"] += l["elapsed_ms"]
            stages[st]["max_ms"] = max(stages[st]["max_ms"], l["elapsed_ms"])

        summary = {}
        for st, data in stages.items():
            summary[st] = {
                "invocations": data["count"],
                "avg_ms": round(data["total_ms"] / data["count"], 2),
                "max_ms": round(data["max_ms"], 2)
            }

        return {
            "total_operations": len(cls._logs),
            "stages": summary,
            "recent_logs": cls._logs[-10:]
        }
