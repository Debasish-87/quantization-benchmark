import os
import platform
import subprocess
import sys
from datetime import datetime, timezone


def gpu_info():
    try:
        import pynvml
        pynvml.nvmlInit()
        out = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            out.append({
                "index": i,
                "name": pynvml.nvmlDeviceGetName(h).decode(),
                "memory_mb": round(mem.total / 1024**2, 1),
                "driver": pynvml.nvmlSystemGetDriverVersion().decode(),
            })
        pynvml.nvmlShutdown()
        return out
    except Exception as e:
        return [{"error": str(e)}]


def environment():
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "gpu": gpu_info(),
        "vllm_version": _version("vllm"),
        "torch_version": _version("torch"),
        "transformers_version": _version("transformers"),
    }


def _version(name):
    try:
        module = __import__(name)
        return getattr(module, "__version__", "unknown")
    except Exception:
        return "not-installed"
