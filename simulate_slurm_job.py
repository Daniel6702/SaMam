#!/usr/bin/env python3

import os
import subprocess
from pathlib import Path
from datetime import datetime
import shlex
import sys

# ------------------------
# Parameters (same as Slurm)
# ------------------------
ITERATIONS = 10000
LEARNING_RATE = 0.0002
PATCH_SIZE = 8
BATCH_SIZE = 8
VALIDATION_INTERVAL = 250

EARLY_STOPPING = 1
PATIENCE = 8
DELTA = 2

APPLY_IDENTITY_LOSS = 1
APPLY_HUBER_LOSS = 0
APPLY_SSIM_LOSS = 0
SSIM_WEIGHT = 2

# ------------------------
# Build run name
# ------------------------
lr_str = f"{LEARNING_RATE:.0e}"
it_str = f"{ITERATIONS // 1000}k"

run_name = (
    f"ps{PATCH_SIZE}_lr{lr_str}_bs{BATCH_SIZE}_it{it_str}_"
    f"id{APPLY_IDENTITY_LOSS}_es{EARLY_STOPPING}_"
    f"hb{APPLY_HUBER_LOSS}_ssim{APPLY_SSIM_LOSS}_ssim_w{SSIM_WEIGHT}"
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ------------------------
# Paths
# ------------------------
output_dir = Path("./OUTPUT") / f"{run_name}_{timestamp}"
checkpoint_file = output_dir / "checkpoint.ckpt"
loss_log_file = output_dir / "loss_log.txt"
evaluation_file = output_dir / "evaluation.txt"
args_file = output_dir / "args.txt"

# ------------------------
# Create venv if missing
# ------------------------
venv_dir = Path(".venv")
python_bin = venv_dir / "bin" / "python"
pip_bin = venv_dir / "bin" / "pip"

if not venv_dir.exists():
    subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)

# ------------------------
# Install dependencies
# ------------------------
subprocess.run([str(pip_bin), "install", "-r", "requirements.txt", "--quiet"], check=True)
subprocess.run([str(pip_bin), "install", "scikit-learn", "--quiet"], check=True)

# ------------------------
# Create output directory
# ------------------------
output_dir.mkdir(parents=True, exist_ok=True)

# ------------------------
# Build argument list
# ------------------------
args = [
    "--content", "./TRAIN/Dataset/MS_COCO/train2017",
    "--style", "./TRAIN/Dataset/wikiart/train",
    "--test-content", "./TRAIN/test_images/content",
    "--test-style", "./TRAIN/test_images/style",
    "--log-dir", "./TRAIN/",
    "--gpus", "0",
    "--patch-size", str(PATCH_SIZE),
    "--output", str(checkpoint_file),
    "--iterations", str(ITERATIONS),
    "--lr", str(LEARNING_RATE),
    "--batch-size", str(BATCH_SIZE),
    "--loss-log", str(loss_log_file),
    "--val-interval", str(VALIDATION_INTERVAL),
    "--patience", str(PATIENCE),
    "--delta", str(DELTA),
    "--quiet",
]

# Conditional flags (exact match to bash logic)
if APPLY_IDENTITY_LOSS == 1:
    args.append("--apply-identity-loss")

if EARLY_STOPPING == 1:
    args.append("--early-stopping")

if APPLY_HUBER_LOSS == 1:
    args.append("--apply-huber-loss")

if APPLY_SSIM_LOSS == 1:
    args.append("--apply-SSIM-loss")

# ------------------------
# Write args.txt (bash-equivalent quoting)
# ------------------------
args_file.write_text(" ".join(shlex.quote(a) for a in args) + "\n")

# ------------------------
# Environment
# ------------------------
env = os.environ.copy()
env["PYTHONPATH"] = f"{os.getcwd()}:{env.get('PYTHONPATH', '')}"

# ------------------------
# Train
# ------------------------
subprocess.run(
    [str(python_bin), "./TRAIN/train_SaMam.py", *args],
    check=True,
    env=env
)

# ------------------------
# Evaluate
# ------------------------
subprocess.run(
    [
        str(python_bin),
        "./TEST/eval.py",
        "--checkpoint", str(checkpoint_file),
        "--output", str(evaluation_file),
    ],
    check=True,
    env=env
)

print("DONE")