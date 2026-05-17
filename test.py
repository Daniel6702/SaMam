#!/usr/bin/env python3

import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# --------------------
# Sweep settings
# --------------------
SSIM_SWEEP = [
    {"weight": 0, "apply_ssim_loss": False, "label": "W0_control_ssim_disabled"},
    {"weight": 10, "apply_ssim_loss": True, "label": "W10"},
    {"weight": 20, "apply_ssim_loss": True, "label": "W20"},
    {"weight": 30, "apply_ssim_loss": True, "label": "W30"},
    {"weight": 40, "apply_ssim_loss": True, "label": "W40"},
    {"weight": 50, "apply_ssim_loss": True, "label": "W50"},
    {"weight": 60, "apply_ssim_loss": True, "label": "W50"},
]

EVAL_START_DELAY_SECONDS = 10


# --------------------
# Arguments
# --------------------
EVAL_FRACTION = 1.0

ITERATIONS = 4000
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
ACTIVATION = "silu"

HUBER_DELTAS = [0.5, 0.1, 0.1]

NOTE_PREFIX = "SSIM_2k_sweep"


# --------------------
# Environment
# --------------------
cwd = os.getcwd()
env = os.environ.copy()
env["PYTHONPATH"] = f"{cwd}:{env.get('PYTHONPATH', '')}"
env["PYTHONHASHSEED"] = "1234"
env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def build_run(ssim_weight: int, apply_ssim_loss: bool, label: str):
    # --------------------
    # Build readable strings
    # --------------------
    lr_str = f"{LEARNING_RATE:.0e}"
    it_str = f"{ITERATIONS // 1000}k"

    id_str = f"id{int(APPLY_IDENTITY_LOSS)}"
    es_str = f"es{int(EARLY_STOPPING)}"
    hb_str = f"hb{int(APPLY_HUBER_LOSS)}"
    ss_str = f"ssim{int(apply_ssim_loss)}"
    ssw_str = f"ssimW{ssim_weight}"
    bt_str = f"bat{int(APPLY_BATCHING)}"
    huber_str = "-".join(str(x) for x in HUBER_DELTAS)

    note = f"{NOTE_PREFIX}_{label}"

    run_name = (
        f"ps{PATCH_SIZE}_lr{lr_str}_bs{BATCH_SIZE}_it{it_str}_"
        f"{id_str}_{es_str}_{hb_str}_hd{huber_str}_"
        f"{ss_str}_{ssw_str}_{ACTIVATION}_{bt_str}_{note}"
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
    eval_args_file = output_dir / "eval_args.txt"
    time_file = output_dir / "time.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------
    # Build train argument list
    # --------------------
    train_args = [
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
        "--checkpoint", "base.ckpt",
        "--ssim-weight", str(ssim_weight),
    ]

    if APPLY_IDENTITY_LOSS:
        train_args.append("--apply-identity-loss")

    if EARLY_STOPPING:
        train_args.append("--early-stopping")

    if APPLY_HUBER_LOSS:
        train_args.append("--apply-huber-loss")

    if apply_ssim_loss:
        train_args.append("--apply-SSIM-loss")

    if APPLY_BATCHING:
        train_args.append("--apply-batching")

    if APPLY_LOW_VRAM:
        train_args.append("--low-vram")

    train_cmd = [
        sys.executable,
        "./TRAIN/train_SaMam.py",
        *train_args,
    ]

    eval_cmd = [
        sys.executable,
        "./TEST/eval.py",
        "--checkpoint", str(checkpoint_file),
        "--output", str(evaluation_file),
        "--pair_fraction", str(EVAL_FRACTION),
    ]

    # Save arguments for reproducibility
    with args_file.open("w") as f:
        f.write(" ".join(shlex.quote(arg) for arg in train_args))
        f.write("\n")

    with eval_args_file.open("w") as f:
        f.write(" ".join(shlex.quote(arg) for arg in eval_cmd))
        f.write("\n")

    return {
        "label": label,
        "run_name": run_name,
        "output_dir": output_dir,
        "checkpoint_file": checkpoint_file,
        "evaluation_file": evaluation_file,
        "train_cmd": train_cmd,
        "eval_cmd": eval_cmd,
        "ssim_weight": ssim_weight,
        "apply_ssim_loss": apply_ssim_loss,
    }


def main():
    eval_processes = []

    try:
        for index, setup in enumerate(SSIM_SWEEP):
            run = build_run(
                ssim_weight=setup["weight"],
                apply_ssim_loss=setup["apply_ssim_loss"],
                label=setup["label"],
            )

            print("=" * 80, flush=True)
            print(f"START TRAINING: {run['label']}", flush=True)
            print(f"SSIM enabled: {run['apply_ssim_loss']}", flush=True)
            print(f"SSIM weight:   {run['ssim_weight']}", flush=True)
            print(f"Output dir:    {run['output_dir']}", flush=True)
            print("=" * 80, flush=True)

            subprocess.run(
                run["train_cmd"],
                check=True,
                env=env,
            )

            print(f"TRAINING DONE: {run['label']}", flush=True)
            print(f"STARTING EVAL ASYNC: {run['label']}", flush=True)

            eval_proc = subprocess.Popen(
                run["eval_cmd"],
                env=env,
            )

            eval_processes.append(
                {
                    "label": run["label"],
                    "output_dir": run["output_dir"],
                    "process": eval_proc,
                }
            )

            # Allow the evaluation to begin before the next training starts.
            if index < len(SSIM_SWEEP) - 1:
                print(
                    f"Waiting {EVAL_START_DELAY_SECONDS} seconds before next training...",
                    flush=True,
                )
                time.sleep(EVAL_START_DELAY_SECONDS)

        print("=" * 80, flush=True)
        print("ALL TRAINING RUNS FINISHED", flush=True)
        print("Waiting for remaining evaluations...", flush=True)
        print("=" * 80, flush=True)

        failed_evals = []

        for item in eval_processes:
            label = item["label"]
            output_dir = item["output_dir"]
            proc = item["process"]

            return_code = proc.wait()

            if return_code == 0:
                print(f"EVAL DONE: {label}", flush=True)
            else:
                print(f"EVAL FAILED: {label} | return code {return_code}", flush=True)
                failed_evals.append((label, output_dir, return_code))

        if failed_evals:
            print("\nSome evaluations failed:", flush=True)
            for label, output_dir, return_code in failed_evals:
                print(
                    f"  - {label}: return code {return_code}, output dir: {output_dir}",
                    flush=True,
                )
            raise SystemExit(1)

        print("DONE", flush=True)

    except KeyboardInterrupt:
        print("\nInterrupted. Terminating active evaluation processes...", flush=True)

        for item in eval_processes:
            proc = item["process"]
            if proc.poll() is None:
                proc.terminate()

        raise


if __name__ == "__main__":
    main()