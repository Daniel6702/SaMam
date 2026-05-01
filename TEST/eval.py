#!/usr/bin/env python3
"""
One-shot evaluation script for SaMam-style checkpoints using the StyleID protocol.

What it does:
1. Loads the checkpoint you pass in.
2. Uses the fixed eval content/style folders.
3. Selects the first N_CONTENT_EVAL content images and first N_STYLE_EVAL style images.
4. Builds the Cartesian product of those subsets:
       every content x every style
   so total pairs = N_CONTENT_EVAL * N_STYLE_EVAL.
5. Writes temporary prepared content/style/output folders with strict 1:1:1 aligned filenames,
   which is what StyleID's evaluation script expects.
6. Runs StyleID's full `evaluation/eval_artfid.py` on those folders.
7. Prints the metrics from StyleID eval_artfid.py and cleans up the temporary files.

Usage:
    python eval_artfid.py /path/to/model.ckpt
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

        # Case 1: this directory is already the SaMam repo root
        if all((candidate / part).exists() for part in ("TEST", "TRAIN", "MODEL")):
            return candidate

        # Case 2: running from a parent directory containing ComputerVisionProject/SaMam
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

# -----------------------------------------------------------------------------
# Fixed eval directories (only checkpoint is passed as CLI input)
# -----------------------------------------------------------------------------

CONTENT_DIR = REPO_ROOT / "TEST" / "eval" / "content"
STYLE_DIR = REPO_ROOT / "TEST" / "eval" / "style"

SAMAM_DEVICE = "cuda:0"
EVAL_DEVICE = "cuda"  # if insufficient VRAM, change to "cpu"

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

# -----------------------------------------------------------------------------
# StyleID protocol constants
# -----------------------------------------------------------------------------
# These are the only numbers you should change to alter the benchmark size.
# Total evaluated stylizations = N_CONTENT_EVAL * N_STYLE_EVAL.
N_CONTENT_EVAL = 20
N_STYLE_EVAL = 40

# StyleID eval settings
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
            "Run StyleID-protocol stylization eval (Cartesian product of selected "
            "content/style subsets) and then call StyleID's eval_artfid.py."
        )
    )
    parser.add_argument(
        "model_ckpt",
        type=str,
        help="Path to the model checkpoint (.ckpt).",
    )
    return parser.parse_args()


def list_files(directory: Path):
    files = [p for p in test_utils.files_in(directory) if p.is_file()]
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

    # Free GPU tensors before any resize work
    del content, style

    # Keep eval alignment with content image resolution
    if output.shape[-2:] != target_size:
        output = F.interpolate(
            output,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

    return output[0]


def load_model(ckpt_path: Path, device: torch.device):
    """
    Supports both checkpoint-loading paths implied by your original test script:
    1. LightningModel.load_from_checkpoint(...)
    2. Manual load of checkpoint['state_dict'] into LightningModel(...).model
    """
    ckpt_path = ckpt_path.resolve()

    # First try the direct Lightning restore path.
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
    ckpt_path: Path,
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

    # Print subprocess output so behavior stays the same
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"StyleID eval_artfid.py failed with exit code {result.returncode}"
        )

    # Extract metric lines
    metric_lines = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("ArtFID:") or stripped.startswith("CFSD:"):
            metric_lines.append(stripped)

    if not metric_lines:
        print("[Warning] Could not find ArtFID/CFSD lines in eval output.")
        return

    metrics_text = "\n".join(metric_lines)

    # Save next to checkpoint, using checkpoint name
    txt_path = ckpt_path.with_suffix(".txt")
    txt_path.write_text(metrics_text + "\n", encoding="utf-8")

    print(f"\nSaved metrics to: {txt_path}")

def select_eval_subset(files, n_required: int, label: str):
    if len(files) < n_required:
        raise ValueError(
            f"Not enough {label} images for evaluation: found {len(files)}, "
            f"but need at least {n_required}."
        )
    return files[:n_required]


def build_eval_pairs(content_files, style_files):
    """
    Build the StyleID-style Cartesian product benchmark:
        [(c0, s0), (c0, s1), ..., (c0, sM-1),
         (c1, s0), (c1, s1), ..., (cN-1, sM-1)]
    """
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
    ckpt_path = Path(args.model_ckpt)

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
    num_pairs = len(eval_pairs)

    print(f"Repo root         : {REPO_ROOT}")
    print(f"Checkpoint        : {ckpt_path.resolve()}")
    print(f"Content dir       : {CONTENT_DIR}")
    print(f"Style dir         : {STYLE_DIR}")
    print(f"Selected contents : {len(selected_content_files)}")
    print(f"Selected styles   : {len(selected_style_files)}")
    print(f"Total pairs       : {num_pairs} (= {N_CONTENT_EVAL} x {N_STYLE_EVAL})")

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

        # Materialize the Cartesian product into 1:1-aligned content/style folders.
        print("\nPreparing temporary eval inputs (Cartesian product benchmark)...")
        for idx, (content_src, style_src) in enumerate(
            tqdm(eval_pairs, total=num_pairs, desc="Preparing")
        ):
            filename = f"{idx:04d}.png"
            save_as_png(content_src, prepared_content_dir / filename)
            save_as_png(style_src, prepared_style_dir / filename)

        print("\nGenerating stylized outputs...")
        with torch.inference_mode():
            for idx in tqdm(range(num_pairs), desc="Stylizing"):
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

        # Run full StyleID evaluation script (ArtFID + CFSD)
        run_styleid_eval(
            style_dir=prepared_style_dir,
            content_dir=prepared_content_dir,
            output_dir=output_dir,
            device_str=styleid_device,
            ckpt_path=ckpt_path,
        )

        print("\nEvaluation complete.")


if __name__ == "__main__":
    main()