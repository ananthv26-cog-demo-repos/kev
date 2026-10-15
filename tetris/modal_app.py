"""Train and evaluate the Tetris policy on Modal. One GPU container per run; metrics and checkpoints land on the
`kev-tetris-runs` volume and `pull` copies them into runs/tetris/.

    uv run modal run tetris/modal_app.py::smoke                                   # a T4, one tiny iteration (~3 min)
    uv run modal run tetris/modal_app.py::train --name rl-0.8b --args "--iters 60 --games 16 --max_pieces 150"
    uv run modal run tetris/modal_app.py::train --name rl-0.8b --detach ...       # keeps running after you disconnect
    uv run modal run tetris/modal_app.py::play --run rl-0.8b/checkpoint --games 20
    uv run modal run tetris/modal_app.py::pull --name rl-0.8b                     # -> runs/tetris/rl-0.8b

Cost: the trainer stamps every metrics row with elapsed seconds and an estimate at Modal's list price for the GPU
(tetris.rl.GPU_HOURLY). `train` refuses to start if --budget would be exceeded at the timeout.
"""
import os
import shlex
import subprocess
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
APP = "kev-tetris"
GPU = os.environ.get("KEV_GPU", "H100")
HF_MOUNT, RUNS_MOUNT = "/hf", "/runs"
HOURLY = {"H100": 3.95, "H200": 4.54, "B200": 6.25, "T4": 0.59, "L4": 0.80, "A10G": 1.10}
CAUSAL_CONV1D = "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu12torch2.8cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"

app = modal.App(APP)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .uv_sync(uv_project_dir=str(ROOT), groups=[], extras=["serve"])
    .uv_pip_install("flash-linear-attention==0.5.2", "triton>=3.7.1")      # Gated DeltaNet kernels for the Qwen3.5 backbones (as modal_app.py)
    .uv_pip_install(CAUSAL_CONV1D, extra_options="--no-deps")
    .env({"HF_HOME": HF_MOUNT, "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
          "TRITON_CACHE_DIR": f"{HF_MOUNT}/triton-cache"})
    .add_local_python_source("kev", "tetris")
)
hf_cache = modal.Volume.from_name("kev-hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("kev-tetris-runs", create_if_missing=True)


def _run(cmd, cwd="/root"):
    print("$", " ".join(shlex.quote(c) for c in cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        sys.stdout.write(line); sys.stdout.flush()
    proc.wait()
    return proc.returncode


@app.function(image=image, gpu=GPU, cpu=4, memory=(32768, 65536), timeout=6 * 3600, retries=0, volumes={HF_MOUNT: hf_cache, RUNS_MOUNT: runs})
def run_train(name, args, gpu):
    out = Path(RUNS_MOUNT) / name
    code = _run([sys.executable, "-m", "tetris.rl", "--device", "cuda", "--gpu", gpu, "--out", str(out), *shlex.split(args)])
    runs.commit()
    return code


@app.function(image=image, gpu=GPU, cpu=4, memory=(32768, 65536), timeout=3600, retries=0, volumes={HF_MOUNT: hf_cache, RUNS_MOUNT: runs})
def run_play(run, args):
    path = run if run.count("/") == 1 and not (Path(RUNS_MOUNT) / run).exists() else str(Path(RUNS_MOUNT) / run)   # a Hub id or a run on the volume
    out = Path(RUNS_MOUNT) / "evals" / run.replace("/", "_")
    out.mkdir(parents=True, exist_ok=True)
    code = _run([sys.executable, "-m", "tetris.play", "--agent", "model", "--device", "cuda", "--run", path, "--out", str(out / "games.jsonl"), *shlex.split(args)])
    runs.commit()
    return code


@app.local_entrypoint()
def smoke(gpu: str = "H100"):
    dtype = "fp32" if gpu == "T4" else "bf16"   # Turing has no bf16
    code = run_train.with_options(gpu=gpu).remote("smoke", f"--iters 1 --games 2 --max_pieces 3 --width 6 --height 8 --eval_games 1 --microbatch 2 --epochs 1 --save_every 0 --dtype {dtype}", gpu)
    print("exit", code)


@app.local_entrypoint()
def train(name: str, args: str = "", gpu: str = GPU, budget: float = 25.0, timeout_hours: float = 4.0):
    worst = HOURLY.get(gpu, 5.0) * timeout_hours
    if worst > budget:
        raise SystemExit(f"{gpu} for {timeout_hours} h could cost ${worst:.0f} > budget ${budget:.0f}; lower --timeout-hours or raise --budget")
    print(f"{name}: {gpu}, at most {timeout_hours} h (~${worst:.0f})", flush=True)
    fn = run_train.with_options(gpu=gpu, timeout=int(timeout_hours * 3600))
    code = fn.remote(name, args, gpu)
    print("exit", code)


@app.local_entrypoint()
def play(run: str, games: int = 20, args: str = "", gpu: str = GPU):
    code = run_play.with_options(gpu=gpu).remote(run, f"--games {games} {args}")
    print("exit", code)


@app.local_entrypoint()
def pull(name: str):
    dest = ROOT / "runs" / "tetris" / name
    dest.mkdir(parents=True, exist_ok=True)
    for entry in runs.listdir(name, recursive=True):
        if entry.type != modal.volume.FileEntryType.FILE: continue
        rel = Path(entry.path).relative_to(name)
        target = dest / rel; target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            for chunk in runs.read_file(entry.path): f.write(chunk)
    print("pulled ->", dest)
