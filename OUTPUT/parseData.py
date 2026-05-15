#!/usr/bin/env python3
"""
Collect evaluation metrics and runtime stats from subdirectories into a CSV.

Expected per-directory layout:
  <dir>/evaluation.txt  – contains ArtFID, FID, LPIPS, CFSD values
  <dir>/runtime.json    – contains iterations_this_run, elapsed_seconds,
                          iterations_per_second (optional)
"""

import csv
import json
import os
import re
import sys
from datetime import timedelta
from pathlib import Path


OUTPUT_CSV = "results.csv"
FIELDNAMES = ["folder", "ArtFID", "FID", "LPIPS", "CFSD",
              "Iterations", "Time HH:MM:SS", "Iterations/s"]


def parse_evaluation(path: Path) -> dict:
    """Extract ArtFID, FID, LPIPS, and CFSD from an evaluation.txt file."""
    text = path.read_text()

    def extract(pattern):
        m = re.search(pattern, text)
        return f"{float(m.group(1)):.3f}" if m else ""

    return {
        "ArtFID": extract(r"ArtFID:\s*([\d.]+)"),
        "FID":    extract(r"(?<![A-Za-z])FID:\s*([\d.]+)"),
        "LPIPS":  extract(r"(?<![A-Za-z])LPIPS:\s*([\d.]+)"),
        "CFSD":   extract(r"CFSD:\s*([\d.]+)"),
    }


def parse_runtime(path: Path) -> dict:
    """Extract iteration count, elapsed time, and speed from a runtime.json file."""
    with path.open() as f:
        data = json.load(f)

    elapsed = int(data["elapsed_seconds"])
    hh_mm_ss = str(timedelta(seconds=elapsed))          # e.g. "0:28:54"
    hh_mm_ss = hh_mm_ss.zfill(8)[:8]                   # pad to "00:28:54"

    return {
        "Iterations":    str(int(data["iterations_this_run"])),
        "Time HH:MM:SS": hh_mm_ss,
        "Iterations/s":  f"{data['iterations_per_second']:.3f}",
    }


def collect(base_dir: Path = Path(".")) -> list[dict]:
    """Walk subdirectories (newest first) and build result rows."""
    subdirs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )

    rows = []
    for d in subdirs:
        eval_file = d / "evaluation.txt"
        if not eval_file.exists():
            continue

        folder = re.sub(r"_\d{8}_\d{6}$", "", d.name)
        row = {"folder": folder}
        row.update(parse_evaluation(eval_file))

        runtime_file = d / "time.json"
        if runtime_file.exists():
            row.update(parse_runtime(runtime_file))
        else:
            row.update({"Iterations": "", "Time HH:MM:SS": "", "Iterations/s": ""})

        rows.append(row)

    return rows


def main():
    base_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    rows = collect(base_dir)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. {len(rows)} rows written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
