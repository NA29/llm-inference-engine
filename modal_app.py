"""Run tests and benchmarks on a cloud A10G via Modal.

Usage:
    modal run modal_app.py --cmd "bench.env"
    modal run modal_app.py --cmd "pytest tests -q"
    modal run modal_app.py --cmd "bench.throughput --num-prompts 1000" --vllm

Whatever the command writes to $RESULTS_DIR is copied back into ./results/,
together with the full stdout log of the run.
"""

import os
import subprocess
import time
from pathlib import Path

import modal

GPU = "A10G"
MODEL_ID = "unsloth/Llama-3.2-1B"  # ungated mirror of meta-llama/Llama-3.2-1B
REPO_DIR = "/root/repo"
RESULTS_DIR = "/root/results"

hf_cache = modal.Volume.from_name("lie-hf-cache", create_if_missing=True)

_ignore = [".git", "results", "**/__pycache__", ".venv", ".pytest_cache"]

engine_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("datasets==3.1.0")
    .env({"HF_HOME": "/hf", "MODEL_ID": MODEL_ID})
    .add_local_dir(".", REPO_DIR, ignore=_ignore)
)

# vLLM pins its own torch, so it lives in a separate image.
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.10.1.1", "transformers==4.56.2", "numpy", "datasets==3.1.0")
    .env({"HF_HOME": "/hf", "MODEL_ID": MODEL_ID})
    .add_local_dir(".", REPO_DIR, ignore=_ignore)
)

app = modal.App("llm-inference-engine")


def _run(cmd: str, git_sha: str, gpu_name: str) -> dict[str, bytes] | None:
    # Modal's "A10G" pool also contains A10s (72 SMs, 150 W vs 80 SMs, 300 W); results are
    # only comparable on one model, so refuse to run on anything else and let the caller retry.
    found = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True).stdout.strip()
    if gpu_name and found != gpu_name:
        print(f"got {found!r}, need {gpu_name!r}; retrying on another machine")
        return None
    os.makedirs(RESULTS_DIR, exist_ok=True)
    env = dict(os.environ, RESULTS_DIR=RESULTS_DIR, GIT_SHA=git_sha, PYTHONUNBUFFERED="1")
    argv = ["python", "-m", *cmd.split()]  # "bench.x ..." or "pytest ..."
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    proc = subprocess.Popen(argv, cwd=REPO_DIR, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)
    proc.wait()
    hf_cache.commit()
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    header = (f"# cmd: {cmd}\n# git_sha: {git_sha}\n# started: {started}\n"
              f"# modal_task_id: {os.environ.get('MODAL_TASK_ID')}\n# exit_code: {proc.returncode}\n"
              f"# nvidia-smi at end of run:\n{smi}\n")
    root = Path(RESULTS_DIR)
    out = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    name = cmd.split()[0].replace("bench.", "").replace("tests.", "")
    out[f"logs/{name}_{started.replace(':', '')}.log"] = (header + "".join(lines)).encode()
    return out


@app.function(gpu=GPU, image=engine_image, volumes={"/hf": hf_cache}, timeout=3 * 3600, single_use_containers=True)
def run_engine(cmd: str, git_sha: str, gpu_name: str):
    return _run(cmd, git_sha, gpu_name)


@app.function(gpu=GPU, image=vllm_image, volumes={"/hf": hf_cache}, timeout=3 * 3600, single_use_containers=True)
def run_vllm(cmd: str, git_sha: str, gpu_name: str):
    return _run(cmd, git_sha, gpu_name)


@app.local_entrypoint()
def main(cmd: str = "bench.env", vllm: bool = False, gpu_name: str = "NVIDIA A10G"):
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
    sha = (sha or "uncommitted") + ("-dirty" if dirty else "")
    fn = run_vllm if vllm else run_engine
    for _ in range(20):
        files = fn.remote(cmd, sha, gpu_name)
        if files is not None:
            break
    else:
        raise RuntimeError(f"no {gpu_name} available")
    for name, data in files.items():
        path = Path("results") / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(f"saved {path}")
