import subprocess
import time
from pathlib import Path

import requests


class VLLMServer:
    def __init__(self, cfg, method):
        self.cfg = cfg
        self.method = method
        self.process = None

    @property
    def host(self):
        return self.cfg["server"]["host"]

    @property
    def port(self):
        return self.cfg["server"]["port"]

    @property
    def base_url(self):
        return f"http://{self.host}:{self.port}"

    def command(self):
        s = self.cfg["server"]
        m = self.method

        cmd = [
            "vllm", "serve",
            m["model"],
            "--served-model-name", self.cfg["model"]["served_name"],
            "--host", self.host,
            "--port", str(self.port),
            "--dtype", str(m.get("dtype", "auto")),
            "--gpu-memory-utilization", str(s["gpu_memory_utilization"]),
            "--max-model-len", str(s["max_model_len"]),
        ]

        q = m.get("quantization")
        if q:
            cmd += ["--quantization", q]

        if s.get("enforce_eager", False):
            cmd.append("--enforce-eager")

        return cmd

    def start(self, log_file):
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        with open(log_file, "w", encoding="utf-8") as log:
            self.process = subprocess.Popen(
                self.command(),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )

        deadline = time.time() + self.cfg["server"]["startup_timeout_seconds"]
        url = f"{self.base_url}/health"

        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"vLLM exited during startup. See {log_file}"
                )
            try:
                r = requests.get(url, timeout=2)
                if r.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(2)

        self.stop()
        raise TimeoutError(f"vLLM startup timed out. See {log_file}")

    def stop(self):
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process = None
