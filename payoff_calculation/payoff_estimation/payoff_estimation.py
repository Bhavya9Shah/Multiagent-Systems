#!/usr/bin/env python3
"""
payoff_estimation/payoff_estimation.py
Master Pipeline Execution for Empirical Payoff Function Estimation.

CS 329: Foundations of AI: Multi-Agent Systems (2026)
Research Project: Click Appeal as a Potential Game: Modeling Creator Incentives,
                  Attention Congestion, and Social Welfare in YouTube
Author: Bhavya Shah (24110324) - Payoff Function Estimation & Statistical Evaluation

Executes:
1. scripts/prepare_data.py
2. scripts/run_regressions.py
3. scripts/generate_plots.py

Generates all datasets, regression models, hypothesis tests, diagnostic plots,
and the machine-readable game-theoretic coefficient payload for teammate Khushi Shah.
"""

import sys
import os
import time
import subprocess
import logging
from pathlib import Path

PAYOFF_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = PAYOFF_DIR / "scripts"
LOGS_DIR = PAYOFF_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "master_pipeline.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("master_pipeline")


def run_stage(script_name, description):
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        sys.exit(1)

    logger.info(f"\n=======================================================")
    logger.info(f"STARTING STAGE: {description} ({script_name})")
    logger.info(f"=======================================================")
    
    start_time = time.time()
    result = subprocess.run([sys.executable, str(script_path)], capture_output=False)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        logger.error(f"Stage failed with return code {result.returncode}: {script_name}")
        sys.exit(result.returncode)
    else:
        logger.info(f"STAGE COMPLETED: {description} in {elapsed:.2f} seconds.")


def main():
    start_total = time.time()
    print("\n" + "=" * 70)
    print(" EMPIRICAL PAYOFF FUNCTION ESTIMATION — MASTER RESEARCH PIPELINE")
    print(" CS 329: Multi-Agent Systems | IIT Gandhinagar (2026)")
    print(" Bhavya Shah (24110324)")
    print("=" * 70 + "\n")

    # Stage 1: Data Preparation & LOCO Peer Construction
    run_stage("prepare_data.py", "Stage 1: Data Preparation & Competitor LOCO Construction")

    # Stage 2: Econometric Estimation & Hypothesis Testing
    run_stage("run_regressions.py", "Stage 2: Econometric Estimation & Game Theory Param Export")

    # Stage 3: Diagnostic Visualizations
    run_stage("generate_plots.py", "Stage 3: Publication-Quality Diagnostic Plot Generation")

    elapsed_total = time.time() - start_total
    print("\n" + "=" * 70)
    print(f" ALL PAYOFF ESTIMATION STAGES COMPLETED SUCCESSFULLY IN {elapsed_total:.2f}s")
    print(f" Deliverables saved to: {PAYOFF_DIR}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
