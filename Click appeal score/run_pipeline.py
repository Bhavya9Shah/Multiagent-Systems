#!/usr/bin/env python3
"""
run_pipeline.py
Master End-to-End Orchestrator for Click Appeal Score Research Pipeline.

Executes all research stages in sequence:
  Stage 2: 01_clean_data.py               -> Inspect & clean raw dataset
  Stage 3: 02_download_thumbnails.py       -> Download all 186 thumbnails & manifest
  Stage 4: 03_extract_title_features.py    -> Linguistic & lexical feature extraction
  Stage 5: 04_extract_thumbnail_features.py-> MediaPipe + FER vision features & visual audit
  Stage 6: 05_calculate_scores.py          -> Winsorization, normalization, composite scores
  Stage 7: 06_creator_analysis.py          -> Creator LOCO aggregation & 10 statistical figures
  Stage 8: 07_validate_results.py          -> Exhaustive mathematical & data integrity validation

Designed for full reproducibility, transparent resumption, and beginner clarity.
"""

import sys
import os
import time
import subprocess
import logging
from pathlib import Path

# Configure logging
BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "run_pipeline.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("run_pipeline")

SCRIPTS = [
    ("Stage 2: Clean Dataset", "scripts/01_clean_data.py"),
    ("Stage 3: Download Thumbnails", "scripts/02_download_thumbnails.py"),
    ("Stage 4: Extract Title Features", "scripts/03_extract_title_features.py"),
    ("Stage 5: Extract Thumbnail Features", "scripts/04_extract_thumbnail_features.py"),
    ("Stage 6: Calculate Scores", "scripts/05_calculate_scores.py"),
    ("Stage 7: Creator & Statistical Analysis", "scripts/06_creator_analysis.py"),
    ("Stage 8: End-to-End Validation", "scripts/07_validate_results.py")
]


def run_pipeline():
    print("\n" + "="*80)
    print("      CLICK APPEAL SCORE — RESEARCH PIPELINE MASTER ORCHESTRATOR")
    print("="*80 + "\n")

    # Select Python executable (prefer virtual environment)
    venv_python = BASE_DIR / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        venv_python = BASE_DIR / ".venv" / "bin" / "python"
    python_cmd = str(venv_python) if venv_python.exists() else sys.executable

    logger.info(f"Using Python runtime: {python_cmd}")

    total_start_time = time.time()
    stage_times = []

    for idx, (stage_name, script_rel_path) in enumerate(SCRIPTS, start=1):
        script_path = BASE_DIR / script_rel_path
        if not script_path.exists():
            logger.error(f"Missing script file: {script_path}")
            sys.exit(1)

        print(f"\n>>> Running [{idx}/{len(SCRIPTS)}] {stage_name} ({script_rel_path})...")
        stage_start = time.time()

        result = subprocess.run([python_cmd, str(script_path)], cwd=str(BASE_DIR))
        elapsed = time.time() - stage_start
        stage_times.append((stage_name, elapsed, result.returncode))

        if result.returncode != 0:
            logger.error(f"FAILURE in {stage_name}! Script exited with code {result.returncode}.")
            print(f"\n[ERROR] Pipeline halted due to error in {stage_name}. Check logs for details.")
            sys.exit(result.returncode)

        logger.info(f"Completed {stage_name} in {elapsed:.2f} seconds.")

    total_elapsed = time.time() - total_start_time

    print("\n" + "="*80)
    print("                       PIPELINE EXECUTION SUMMARY")
    print("="*80)
    for name, elapsed, code in stage_times:
        status_text = "SUCCESS" if code == 0 else f"FAILED ({code})"
        print(f"  {name:<42} : {status_text:<10} ({elapsed:.2f}s)")
    print("-" * 80)
    print(f"  Total Pipeline Execution Time              : {total_elapsed:.2f}s ({total_elapsed/60:.2f} mins)")
    print("="*80 + "\n")
    logger.info("All pipeline stages executed and validated successfully.")


if __name__ == "__main__":
    run_pipeline()
