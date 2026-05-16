#!/usr/bin/env python3

import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path



# --------------------
# Arguments
# --------------------
EVAL_FRACTION = 1.0

ITERATIONS = 5000
LEARNING_RATE = 0.0002
PATCH_SIZE = 8
BATCH_SIZE = 4
VALIDATION_INTERVAL = 250
ACCUMULATE_GRAD_BATCHES = 2

EARLY_STOPPING = False
PATIENCE = 8
DELTA = 1

APPLY_LOW_VRAM = False
APPLY_IDENTITY_LOSS = True
APPLY_BATCHING = True
APPLY_HUBER_LOSS = False
APPLY_SSIM_LOSS = False
SSIM_WEIGHT = 2
ACTIVATION = "silu"

HUBER_DELTAS = [0.5, 0.1, 0.1]

NOTE = "TEST10"

# --------------------
# Build readable strings
# --------------------
lr_str = f"{LEARNING_RATE:.0e}"
it_str = f"{ITERATIONS // 1000}k"

id_str = f"id{int(APPLY_IDENTITY_LOSS)}"
es_str = f"es{int(EARLY_STOPPING)}"
hb_str = f"hb{int(APPLY_HUBER_LOSS)}"
ss_str = f"ssim{int(APPLY_SSIM_LOSS)}"
ssw_str = f"ssimW{SSIM_WEIGHT}"
bt_str = f"bat{int(APPLY_BATCHING)}"
huber_str = "-".join(str(x) for x in HUBER_DELTAS)

run_name = (
    f"ps{PATCH_SIZE}_lr{lr_str}_bs{BATCH_SIZE}_it{it_str}_"
    f"{id_str}_{es_str}_{hb_str}_hd{huber_str}_"
    f"{ss_str}_{ssw_str}_{ACTIVATION}_{bt_str}_{NOTE}"
)


# --------------------
# Files and folders
# --------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

output_dir = Path("./OUTPUT") / f"{run_name}_{timestamp}"
checkpoint_file = output_dir / "checkpoint.ckpt"
loss_log_file = output_dir / "loss_log.txt"
evaluation_file = output_dir / "evaluation.txt"
args_file = output_dir / "args.txt"
time_file = output_dir / "time.json"

output_dir.mkdir(parents=True, exist_ok=True)


# --------------------
# Build argument list
# --------------------
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
    "--activation", ACTIVATION,
    "--quiet",
    "--time-log", str(time_file),
    "--huber-deltas", *map(str, HUBER_DELTAS),
    "--seed", "1234",
    "--accumulate-grad-batches", str(ACCUMULATE_GRAD_BATCHES),
]

if APPLY_IDENTITY_LOSS:
    args.append("--apply-identity-loss")

if EARLY_STOPPING:
    args.append("--early-stopping")

if APPLY_HUBER_LOSS:
    args.append("--apply-huber-loss")

if APPLY_SSIM_LOSS:
    args.append("--apply-SSIM-loss")

if APPLY_BATCHING:
    args.append("--apply-batching")

if APPLY_LOW_VRAM:
    args.append("--low-vram")

# Save arguments for reproducibility
with args_file.open("w") as f:
    f.write(" ".join(shlex.quote(arg) for arg in args))
    f.write("\n")


# Match: export PYTHONPATH="$PWD:$PYTHONPATH"
cwd = os.getcwd()
env = os.environ.copy()
env["PYTHONPATH"] = f"{cwd}:{env.get('PYTHONPATH', '')}"
env["PYTHONHASHSEED"] = "1234"
env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# --------------------
# Train
# --------------------
subprocess.run(
    ["python", "./TRAIN/train_SaMam.py", *args],
    check=True,
    env=env,
)


# --------------------
# Evaluate
# --------------------
subprocess.run(
    [
        "python",
        "./TEST/eval.py",
        "--checkpoint", str(checkpoint_file),
        "--output", str(evaluation_file),
        "--pair_fraction", str(EVAL_FRACTION)
    ],
    check=True,
    env=env,
)


print("DONE")