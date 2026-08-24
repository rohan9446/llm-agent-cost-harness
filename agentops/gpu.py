"""
GPU telemetry sampler.

Allocated GPU time is the cost basis, but allocated time says nothing about
whether the GPU was doing anything.

**What `utilization.gpu` actually means.** NVML defines it as the percent of
the sample period during which *one or more* kernels was executing. It is an
occupancy-blind duty cycle: a single tiny kernel keeping one SM busy for the
whole period reads 100%, identically to a kernel saturating all 108. So a
`busy_fraction` of 1.0 is evidence that the device was never idle -- and is NOT
evidence that there is no headroom. Decode at low batch size is the textbook
case: memory-bandwidth-bound, most SMs starved, `utilization.gpu` pinned at
100%.

Read this field as "was the GPU ever idle", and read achieved tokens/s against
what the hardware can do for the real headroom number. Treating 100% here as
"saturated" would retire the concurrency work on a misreading.

Samples nvidia-smi in a background thread for the duration of a run. If
nvidia-smi is absent the sampler degrades to a no-op and says so, rather than
reporting zeros that would read as a measurement.
"""

from __future__ import annotations

import os
import shutil
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

QUERY = "index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,clocks.current.sm"
FIELDS = ["index", "util_gpu", "util_mem", "mem_used_mb", "mem_total_mb",
          "temp_c", "power_w", "sm_clock_mhz"]


@dataclass
class GpuSampler:
    """Samples only the devices under test.

    The lab box has three GPUs and is shared. A bare `nvidia-smi` query returns
    all three, so gpu.json would carry someone else's workload while the cost
    model charges for one -- and the report would print three GPUs for a
    single-GPU run. `devices` defaults to CUDA_VISIBLE_DEVICES, which is what
    the run was actually given.
    """
    devices: str | None = None
    interval_s: float = 0.5
    samples: list[dict[str, Any]] = field(default_factory=list)
    available: bool = True
    reason: str = ""
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def __post_init__(self) -> None:
        if shutil.which("nvidia-smi") is None:
            self.available = False
            self.reason = "nvidia-smi not found; GPU telemetry unavailable"
        if self.devices is None:
            self.devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() or None
        # nvidia-smi -i indexes physical devices; CUDA_VISIBLE_DEVICES is a
        # physical-index list too, so it can be passed straight through.
        self._sel = ["-i", self.devices] if self.devices else []

    def _sample_once(self) -> None:
        out = subprocess.run(
            ["nvidia-smi", *self._sel,
             f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        now = time.time()
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != len(FIELDS):
                continue
            row: dict[str, Any] = {"ts": now}
            for k, v in zip(FIELDS, parts):
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = None
            self.samples.append(row)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample_once()
            except Exception:  # noqa: BLE001 - telemetry must not kill a run
                pass
            self._stop.wait(self.interval_s)

    def start(self) -> "GpuSampler":
        if not self.available:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> "GpuSampler":
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return self

    def __enter__(self) -> "GpuSampler":
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.stop()

    def summary(self) -> dict[str, Any]:
        if not self.available:
            return {"available": False, "reason": self.reason}
        if not self.samples:
            return {"available": True, "n_samples": 0,
                    "reason": "sampler ran but collected nothing"}

        by_gpu: dict[Any, list[dict]] = {}
        for s in self.samples:
            by_gpu.setdefault(s.get("index"), []).append(s)

        out: dict[str, Any] = {"available": True, "n_samples": len(self.samples),
                               "interval_s": self.interval_s,
                               "devices_under_test": self.devices or "all (unfiltered)",
                               "gpus": {}}
        for idx, rows in sorted(by_gpu.items(), key=lambda kv: kv[0] or 0):
            def col(k: str) -> list[float]:
                return [r[k] for r in rows if r.get(k) is not None]

            util = col("util_gpu")
            mem = col("mem_used_mb")
            out["gpus"][int(idx) if idx is not None else "?"] = {
                "util_gpu_mean": statistics.fmean(util) if util else None,
                "util_gpu_p50": statistics.median(util) if util else None,
                "util_gpu_p95": _pct(util, 0.95),
                "util_gpu_max": max(util) if util else None,
                "busy_fraction": (sum(1 for u in util if u > 5) / len(util)) if util else None,
                "mem_used_mb_max": max(mem) if mem else None,
                "mem_total_mb": rows[-1].get("mem_total_mb"),
                "temp_c_max": max(col("temp_c") or [0]) or None,
                "power_w_mean": statistics.fmean(col("power_w")) if col("power_w") else None,
                "sm_clock_mhz_mean": statistics.fmean(col("sm_clock_mhz")) if col("sm_clock_mhz") else None,
            }
        return out


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    i = min(len(s) - 1, int(round(q * (len(s) - 1))))
    return s[i]
