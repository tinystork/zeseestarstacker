import os
import threading
import time
import statistics
import logging
from pathlib import Path

try:
    import psutil

    _PSUTIL_OK = True
except Exception:  # ImportError or others
    _PSUTIL_OK = False

log = logging.getLogger(__name__)


class CpuIoAutoTuner:
    def __init__(
        self, stacker: "SeestarQueuedStacker", duration: int = 180, target: float = 0.75
    ) -> None:
        """Auto tune CPU thread usage for a SeestarQueuedStacker."""
        self.stacker = stacker
        self.duration = int(duration)
        self.target = float(target)
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

        out = Path(getattr(stacker, "output_folder", ""))
        self.disk_dev = None
        if _PSUTIL_OK and out.drive:
            for part in psutil.disk_partitions():
                if out.drive.lower().startswith(part.device.lower()):
                    self.disk_dev = part.device
                    break

    # ------------------------------------------------------------------
    def start(self) -> None:
        if not _PSUTIL_OK:
            log.warning("AutoTune d\u00e9sactiv\u00e9 : psutil manquant.")
            return
        self._t.start()

    def stop(self) -> None:
        self._stop.set()
        if self._t.is_alive():
            self._t.join(timeout=3)

    # ------------------------------------------------------------------
    def _run(self) -> None:
        t0 = time.time()
        cpu_samples: list[float] = []
        io_samples: list[int] = []
        io_prev = (
            psutil.disk_io_counters(perdisk=True).get(self.disk_dev)
            if self.disk_dev
            else None
        )

        while not self._stop.is_set() and time.time() - t0 < self.duration:
            cpu_samples.append(psutil.cpu_percent(interval=1) / 100.0)
            io_now = (
                psutil.disk_io_counters(perdisk=True).get(self.disk_dev)
                if self.disk_dev
                else None
            )
            if io_now and io_prev:
                io_samples.append(
                    (io_now.read_bytes + io_now.write_bytes)
                    - (io_prev.read_bytes + io_prev.write_bytes)
                )
            io_prev = io_now

        if not cpu_samples:
            return
        avg_cpu = statistics.mean(cpu_samples)
        frac = getattr(self.stacker, "thread_fraction", 0.5)

        if avg_cpu < 0.60:
            new_frac = min(frac * 1.5, 0.75)
        elif avg_cpu > 0.85:
            new_frac = max(frac * 0.8, 0.25)
        else:
            return

        if abs(new_frac - frac) < 0.05:
            return

        log.info(
            f"[AutoTune] CPU {avg_cpu:.0%}  \u2192 ajustement threads {frac:.2f} \u2192 {new_frac:.2f}"
        )
        try:
            self.stacker.thread_fraction = new_frac
            self.stacker._configure_global_threads(new_frac)
            self.stacker.max_reproj_workers = max(1, int(os.cpu_count() * new_frac))
        except Exception as e:  # pragma: no cover - log only
            log.error("AutoTune : \u00e9chec de l'application : %s", e)
