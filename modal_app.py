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


def _run(cmd: str, git_sha: str) -> dict[str, bytes]:
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


@app.function(gpu=GPU, image=engine_image, volumes={"/hf": hf_cache}, timeout=3 * 3600)
def run_engine(cmd: str, git_sha: str) -> dict[str, bytes]:
    return _run(cmd, git_sha)


@app.function(gpu=GPU, image=vllm_image, volumes={"/hf": hf_cache}, timeout=3 * 3600)
def run_vllm(cmd: str, git_sha: str) -> dict[str, bytes]:
    return _run(cmd, git_sha)


@app.local_entrypoint()
def main(cmd: str = "bench.env", vllm: bool = False):
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
    sha = (sha or "uncommitted") + ("-dirty" if dirty else "")
    files = (run_vllm if vllm else run_engine).remote(cmd, sha)
    for name, data in files.items():
        path = Path("results") / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(f"saved {path}")
