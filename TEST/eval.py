#!/usr/bin/env python3
"""
One-shot evaluation script for SaMam-style checkpoints using the StyleID protocol.

Usage:
    python eval_artfid.py --checkpoint /path/to/model.ckpt --output /path/to/results.txt
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import random, numpy as np

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)

# -----------------------------------------------------------------------------
# Resolve repo root so imports work regardless of where this file is placed/run.
# -----------------------------------------------------------------------------

def find_repo_root() -> Path:
    candidates = []

    for base in [Path(__file__).resolve().parent, Path.cwd().resolve()]:
        candidates.extend([base, *base.parents])

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)

        if all((candidate / part).exists() for part in ("TEST", "TRAIN", "MODEL")):
            return candidate

        nested = candidate / "ComputerVisionProject" / "SaMam"
        if all((nested / part).exists() for part in ("TEST", "TRAIN", "MODEL")):
            return nested

    raise RuntimeError(
        "Could not find the SaMam repo root. Expected a directory containing "
        "'TEST', 'TRAIN', and 'MODEL'."
    )


REPO_ROOT = find_repo_root()
sys.path.insert(0, str(REPO_ROOT))

from TEST import test_utils  # noqa: E402
from TRAIN.lightning_module.lightningmodel import LightningModel  # noqa: E402


STYLEID_ROOT = REPO_ROOT / "external" / "StyleID"
STYLEID_EVAL_SCRIPT = STYLEID_ROOT / "evaluation" / "eval_artfid.py"

CONTENT_DIR = REPO_ROOT / "TEST" / "eval" / "content"
STYLE_DIR = REPO_ROOT / "TEST" / "eval" / "style"

SAMAM_DEVICE = "cuda:0"
EVAL_DEVICE = "cuda"

DEFAULT_STYLE_SIZE = 256
DEFAULT_MODEL_ARGS = {
    "nVSSMs": 2,
    "nSAVSSMs": 2,
    "nSAVSSGs": 2,
    "embed_dim": 256,
    "patch_size": 8,
    "representation_dim": 64,
    "d_state": 16,
    "expand": 2.0,
    "compress_ratio": 8,
    "squeeze_factor": 8,
    "mamba_from_trion": 1,
}

N_CONTENT_EVAL = 20
N_STYLE_EVAL = 40

STYLEID_BATCH_SIZE = 1
STYLEID_NUM_WORKERS = 8
STYLEID_CONTENT_METRIC = "lpips"
STYLEID_MODE = "art_fid_inf"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run StyleID-protocol stylization eval using a SaMam checkpoint "
            "and save the resulting metrics to an output file."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Path to the model checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Path to the output text file where metrics will be saved.",
    )
    parser.add_argument(
        "--pair_fraction",
        default=1.0,
        type=float,
        help=(
            "Fraction of image pairs to evaluate. "
            "Default is 1.0, which evaluates the full current benchmark. "
            "For faster evaluation, use e.g. 0.2 to evaluate 20%% of pairs."
        ),
    )
    return parser.parse_args()

def select_pair_fraction(eval_pairs, pair_fraction: float):
    if pair_fraction <= 0.0 or pair_fraction > 1.0:
        raise ValueError(
            f"--pair_fraction must be in the range (0, 1], got {pair_fraction}."
        )

    n_pairs = max(1, int(round(len(eval_pairs) * pair_fraction)))
    return eval_pairs[:n_pairs]

def list_files(directory: Path):
    files = sorted([p for p in test_utils.files_in(directory) if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No files found in: {directory}")
    return files


def save_as_png(src: Path, dst: Path):
    img = Image.open(src).convert("RGB")
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, format="PNG")


def stylize_image(model, content_file: Path, style_file: Path, style_size: int):
    device = next(model.parameters()).device

    content_pil = test_utils.load(content_file)
    style_pil = test_utils.load(style_file)

    content = test_utils.content_transforms()(content_pil).unsqueeze(0).to(device)
    style = test_utils.style_transforms(style_size)(style_pil).unsqueeze(0).to(device)

    target_size = content.shape[-2:]

    output = model(content, style)
    output = output.detach().cpu()

    del content, style

    if output.shape[-2:] != target_size:
        output = F.interpolate(
            output,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

    return output[0]


def load_model(ckpt_path: Path, device: torch.device):
    ckpt_path = ckpt_path.resolve()

    try:
        model = LightningModel.load_from_checkpoint(
            checkpoint_path=str(ckpt_path),
            map_location=device,
        )
        model = model.to(device)
        model.eval()
        return model
    except Exception as lightning_exc:
        print(
            f"[Info] LightningModel.load_from_checkpoint failed, "
            f"falling back to manual state_dict loading.\n"
            f"Reason: {lightning_exc}"
        )

    checkpoint = torch.load(ckpt_path, map_location=device)

    if "state_dict" not in checkpoint:
        raise RuntimeError(
            "Fallback loader expected checkpoint['state_dict'], but it was not found."
        )

    model = LightningModel(**DEFAULT_MODEL_ARGS)

    raw_state_dict = checkpoint["state_dict"]
    cleaned_state_dict = {}

    for key, value in raw_state_dict.items():
        if key.startswith("loss_func"):
            continue

        if key.startswith("model."):
            cleaned_key = key[len("model."):]
        else:
            cleaned_key = key

        cleaned_state_dict[cleaned_key] = value

    load_errors = []

    target_modules = []
    if hasattr(model, "model"):
        target_modules.append(("model.model", model.model))
    target_modules.append(("model", model))

    loaded = False
    for target_name, target_module in target_modules:
        try:
            target_module.load_state_dict(cleaned_state_dict, strict=True)
            loaded = True
            print(f"[Info] Loaded checkpoint into {target_name}.")
            break
        except Exception as exc:
            load_errors.append(f"{target_name}: {exc}")

    if not loaded:
        raise RuntimeError(
            "Could not load checkpoint with fallback loader.\n"
            + "\n".join(load_errors)
        )

    model = model.to(device)
    model.eval()
    return model


def run_styleid_eval(
    style_dir: Path,
    content_dir: Path,
    output_dir: Path,
    device_str: str,
    output_path: Path,
):
    if not STYLEID_EVAL_SCRIPT.exists():
        raise FileNotFoundError(
            f"StyleID eval script not found: {STYLEID_EVAL_SCRIPT}"
        )

    cmd = [
        sys.executable,
        "-u",
        str(STYLEID_EVAL_SCRIPT),
        "--batch_size",
        str(STYLEID_BATCH_SIZE),
        "--num_workers",
        str(STYLEID_NUM_WORKERS),
        "--content_metric",
        STYLEID_CONTENT_METRIC,
        "--mode",
        STYLEID_MODE,
        "--device",
        device_str,
        "--sty",
        str(style_dir),
        "--cnt",
        str(content_dir),
        "--tar",
        str(output_dir),
    ]

    print("\nRunning StyleID evaluation:")
    print(" ".join(cmd))
    print()

    env = dict(os.environ)
    pythonpath_entries = [
        str(STYLEID_ROOT),
        str(STYLEID_ROOT / "evaluation"),
    ]
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = ":".join(pythonpath_entries)

    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"StyleID eval_artfid.py failed with exit code {result.returncode}"
        )

    metric_lines = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("ArtFID:") or stripped.startswith("CFSD:"):
            metric_lines.append(stripped)

    if not metric_lines:
        print("[Warning] Could not find ArtFID/CFSD lines in eval output.")
        return

    metrics_text = "\n".join(metric_lines)

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(metrics_text + "\n", encoding="utf-8")

    print(f"\nSaved metrics to: {output_path}")


def select_eval_subset(files, n_required: int, label: str):
    if len(files) < n_required:
        raise ValueError(
            f"Not enough {label} images for evaluation: found {len(files)}, "
            f"but need at least {n_required}."
        )
    return files[:n_required]


def build_eval_pairs(content_files, style_files):
    pairs = []
    for content_file in content_files:
        for style_file in style_files:
            pairs.append((content_file, style_file))
    return pairs


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    output_path = Path(args.output)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if not CONTENT_DIR.exists():
        raise FileNotFoundError(f"Content directory not found: {CONTENT_DIR}")

    if not STYLE_DIR.exists():
        raise FileNotFoundError(f"Style directory not found: {STYLE_DIR}")

    all_content_files = list_files(CONTENT_DIR)
    all_style_files = list_files(STYLE_DIR)

    selected_content_files = select_eval_subset(
        all_content_files, N_CONTENT_EVAL, "content"
    )
    selected_style_files = select_eval_subset(
        all_style_files, N_STYLE_EVAL, "style"
    )

    eval_pairs = build_eval_pairs(selected_content_files, selected_style_files)
    full_num_pairs = len(eval_pairs)

    #eval_pairs = select_pair_fraction(eval_pairs, args.pair_fraction)
    eval_pairs = select_pair_fraction(eval_pairs, 1.0)
    num_pairs = len(eval_pairs)

    print(f"Repo root         : {REPO_ROOT}")
    print(f"Checkpoint        : {ckpt_path.resolve()}")
    print(f"Output file       : {output_path.resolve()}")
    print(f"Content dir       : {CONTENT_DIR}")
    print(f"Style dir         : {STYLE_DIR}")
    print(f"Selected contents : {len(selected_content_files)}")
    print(f"Selected styles   : {len(selected_style_files)}")
    print(f"Pair fraction     : {args.pair_fraction}")
    print(
        f"Total pairs       : {num_pairs} / {full_num_pairs} "
        f"(full = {N_CONTENT_EVAL} x {N_STYLE_EVAL})"
    )

    device = torch.device(SAMAM_DEVICE if torch.cuda.is_available() else "cpu")
    styleid_device = EVAL_DEVICE if torch.cuda.is_available() else "cpu"

    print(f"SaMam device      : {device}")
    print(f"Eval device       : {styleid_device}")

    model = load_model(ckpt_path, device)

    with tempfile.TemporaryDirectory(prefix="artfid_eval_") as tmpdir:
        tmpdir = Path(tmpdir)
        prepared_content_dir = tmpdir / "content"
        prepared_style_dir = tmpdir / "style"
        output_dir = tmpdir / "output"

        prepared_content_dir.mkdir(parents=True, exist_ok=True)
        prepared_style_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\nPreparing temporary eval inputs (Cartesian product benchmark)...")
        for idx, (content_src, style_src) in enumerate(eval_pairs):
            filename = f"{idx:04d}.png"
            save_as_png(content_src, prepared_content_dir / filename)
            save_as_png(style_src, prepared_style_dir / filename)

        print("\nGenerating stylized outputs...")
        with torch.inference_mode():
            for idx in range(num_pairs):
                filename = f"{idx:04d}.png"
                content_file = prepared_content_dir / filename
                style_file = prepared_style_dir / filename

                output = stylize_image(
                    model=model,
                    content_file=content_file,
                    style_file=style_file,
                    style_size=DEFAULT_STYLE_SIZE,
                )
                test_utils.save(output, output_dir / filename)
                del output

                if device.type == "cuda" and (idx + 1) % 8 == 0:
                    torch.cuda.empty_cache()

        if device.type == "cuda":
            model = model.cpu()
        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        run_styleid_eval(
            style_dir=prepared_style_dir,
            content_dir=prepared_content_dir,
            output_dir=output_dir,
            device_str=styleid_device,
            output_path=output_path,
        )

        print("\nEvaluation complete.")


if __name__ == "__main__":
    main()