"""Shared benchmark helpers: environment capture and result saving.

Every result file embeds the hardware/software environment it was produced on
(nvidia-smi, driver, library versions, git commit, Modal task id), so the numbers
in the README can be traced back to a concrete run on a concrete GPU.
"""

import json
import os
import platform
import subprocess
import time
from pathlib import Path

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
MODEL_ID = os.environ.get("MODEL_ID", "unsloth/Llama-3.2-1B")


def _cmd(args: list[str]) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def collect_env() -> dict:
    env = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": os.environ.get("GIT_SHA", _cmd(["git", "rev-parse", "--short", "HEAD"])),
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "cpu": _cmd(["sh", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"]),
        "gpu_query": _cmd(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,"
                           "clocks.max.sm,clocks.max.mem,power.limit", "--format=csv"]),
        "nvidia_smi": _cmd(["nvidia-smi"]),
    }
    for mod in ("torch", "triton", "transformers", "vllm"):
        try:
            env[mod] = __import__(mod).__version__
        except ImportError:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            env["cuda"] = torch.version.cuda
            env["device"] = {"name": p.name, "total_memory_gb": round(p.total_memory / 2**30, 2),
                             "sm_count": p.multi_processor_count, "capability": f"{p.major}.{p.minor}"}
    except ImportError:
        pass
    return env


def save_result(name: str, result: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps({"env": collect_env(), **result}, indent=2))
    print(f"wrote {path}")
    return path
