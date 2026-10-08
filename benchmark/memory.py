import time
import threading


class GPUMonitor:
    def __init__(self, interval=0.25):
        self.interval = interval
        self._stop = threading.Event()
        self.samples = []
        self.thread = None

    def _run(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            while not self._stop.is_set():
                row = []
                for i in range(pynvml.nvmlDeviceGetCount()):
                    h = pynvml.nvmlDeviceGetHandleByIndex(i)
                    m = pynvml.nvmlDeviceGetMemoryInfo(h)
                    row.append({
                        "gpu": i,
                        "used_mb": m.used / 1024**2,
                        "total_mb": m.total / 1024**2,
                    })
                self.samples.append((time.time(), row))
                time.sleep(self.interval)
            pynvml.nvmlShutdown()
        except Exception:
            return

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self._stop.set()
        if self.thread:
            self.thread.join(timeout=2)

    def summary(self):
        if not self.samples:
            return {}
        peak = {}
        for _, rows in self.samples:
            for r in rows:
                peak[r["gpu"]] = max(peak.get(r["gpu"], 0), r["used_mb"])
        return {
            f"gpu_{gpu}_peak_vram_mb": round(mb, 1)
            for gpu, mb in peak.items()
        }
