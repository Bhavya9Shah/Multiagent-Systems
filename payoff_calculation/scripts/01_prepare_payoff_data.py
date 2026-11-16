#!/usr/bin/env python3
"""
payoff_estimation/scripts/01_prepare_payoff_data.py
Phase 2 - Step 1: Data Ingestion, Integrity Validation, and Feature Construction.

This script executes Step 1 of the empirical payoff estimation pipeline:
1. Loads the finalized 809-video Click Appeal Score dataset and cleaned metadata.
2. Performs a rigorous 1:1 merge verification on video_id.
3. Standardizes channel-level subscriber counts (S_i = channel median).
4. Constructs core dependent, explanatory, and control variables:
   - log(views)
   - Click appeal score x and quadratic x^2
   - log(subscribers)
   - log(duration_sec)
   - age_days and log(age_days)
   - log(tags_count + 1) [for robustness check R-Tags]
5. Validates non-negativity, missingness, and boundary conditions.
6. Exports:
   - payoff_estimation/data/processed_payoff_data.csv
   - payoff_estimation/data/data_dictionary.csv
   - payoff_estimation/logs/01_prepare_payoff_data.log
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# Setup directories
SCRIPT_DIR = Path(__file__).resolve().parent
PAYOFF_DIR = SCRIPT_DIR.parent
REPO_ROOT = PAYOFF_DIR.parent
DATA_DIR = PAYOFF_DIR / "data"
LOGS_DIR = PAYOFF_DIR / "logs"

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
LOG_FILE = LOGS_DIR / "01_prepare_payoff_data.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("01_prepare_payoff_data")


def locate_input_files():
    """Locate video_scores.csv and cleaned_videos.csv dynamically across repository paths."""
    score_candidates = [
        REPO_ROOT / "Click appeal score" / "scores" / "video_scores.csv",
        REPO_ROOT / "click_appeal_score" / "scores" / "video_scores.csv",
        REPO_ROOT / "scores" / "video_scores.csv"
    ]
    clean_candidates = [
        REPO_ROOT / "Click appeal score" / "data" / "processed" / "cleaned_videos.csv",
        REPO_ROOT / "click_appeal_score" / "data" / "processed" / "cleaned_videos.csv",
        REPO_ROOT / "data" / "processed" / "cleaned_videos.csv"
    ]

    score_path = next((p for p in score_candidates if p.exists()), None)
    clean_path = next((p for p in clean_candidates if p.exists()), None)

    if not score_path:
        logger.error(f"Cannot find video_scores.csv in: {[str(p) for p in score_candidates]}")
        sys.exit(1)
    if not clean_path:
        logger.error(f"Cannot find cleaned_videos.csv in: {[str(p) for p in clean_candidates]}")
        sys.exit(1)

    logger.info(f"Using Score CSV: {score_path}")
    logger.info(f"Using Clean Metadata CSV: {clean_path}")
    return score_path, clean_path


def run_data_preparation():
    logger.info("=" * 70)
    logger.info("STEP 1: DATA INGESTION, AUDIT & FEATURE PREPARATION")
    logger.info("=" * 70)

    score_path, clean_path = locate_input_files()

    df_scores = pd.read_csv(score_path)
    df_clean = pd.read_csv(clean_path)

    logger.info(f"Loaded scores: {df_scores.shape[0]} rows, {df_scores.shape[1]} cols")
    logger.info(f"Loaded clean metadata: {df_clean.shape[0]} rows, {df_clean.shape[1]} cols")

    # 1. Integrity Audit & 1:1 Merge Verification
    n_unique_scores = df_scores["video_id"].nunique()
    n_unique_clean = df_clean["video_id"].nunique()

    assert n_unique_scores == 809, f"Expected 809 unique scores, got {n_unique_scores}"
    assert n_unique_clean == 809, f"Expected 809 unique clean videos, got {n_unique_clean}"

    scores_set = set(df_scores["video_id"])
    clean_set = set(df_clean["video_id"])

    diff_1 = scores_set - clean_set
    diff_2 = clean_set - scores_set

    if diff_1 or diff_2:
        logger.error(f"Mismatch in video_id! In scores but not clean: {len(diff_1)}. In clean but not scores: {len(diff_2)}.")
        sys.exit(1)
    logger.info("âœ“ Video ID 1:1 match verified: Exactly 809 overlapping unique videos.")

    # 2. Merge datasets cleanly
    # Use df_scores as primary, pulling any auxiliary columns from df_clean if needed
    merged = df_scores.copy()
    if "thumbnail" in df_clean.columns and "thumbnail" not in merged.columns:
        merged = merged.merge(df_clean[["video_id", "thumbnail"]], on="video_id", how="left")

    # 3. Standardize Channel Subscriber Counts
    # Slight scrapetime drift observed in 15 channels (e.g. 6.91M vs 6.93M).
    # We assign channel-level median subscribers S_i to represent the persistent channel scale.
    sub_medians = merged.groupby("channel_name")["subscribers"].median().to_dict()
    merged["subscribers_channel"] = merged["channel_name"].map(sub_medians)
    merged["subscribers_raw"] = merged["subscribers"]

    # 4. Construct Core Econometric Variables
    # Dependent Variable: log(views)
    min_views = merged["views"].min()
    logger.info(f"Minimum view count in sample: {min_views}")
    assert min_views > 0, "Non-positive views detected!"
    merged["log_views"] = np.log(merged["views"])

    # Own Appeal: x and x^2
    merged["x"] = merged["score_weighted"].astype(float)
    merged["x2"] = merged["x"] ** 2

    # Creator Scale: log(S_i)
    merged["log_subscribers"] = np.log(merged["subscribers_channel"])

    # Duration: log(duration_sec)
    assert (merged["duration_sec"] > 0).all(), "Non-positive duration detected!"
    merged["log_duration"] = np.log(merged["duration_sec"])

    # Video Age: age_days and log(age_days)
    assert (merged["age_days"] > 0).all(), "Non-positive age detected!"
    merged["age_days"] = merged["age_days"].astype(float)
    merged["log_age"] = np.log(merged["age_days"])

    # Tag Count: log(tags_count + 1) for robustness check
    assert (merged["tags_count"] >= 0).all(), "Negative tags count detected!"
    merged["log_tags_plus1"] = np.log(merged["tags_count"] + 1)

    # Date parsing
    merged["published_at_dt"] = pd.to_datetime(merged["published_at"])
    merged["year_month"] = merged["published_at_dt"].dt.to_period("M").astype(str)

    # 5. Define Subscriber Tiers (for stratified partitioning)
    def assign_tier(subs):
        if subs < 5_000_000:
            return "Tier_1_Low (<5M)"
        elif subs <= 20_000_000:
            return "Tier_2_Mid (5M-20M)"
        else:
            return "Tier_3_Mega (>20M)"

    merged["subscriber_tier"] = merged["subscribers_channel"].apply(assign_tier)

    # Log summary statistics
    logger.info("-" * 50)
    logger.info("SUMMARY STATISTICS OF PREPARED VARIABLES (N = 809):")
    cols_to_summarize = [
        "views", "log_views", "x", "x2", "subscribers_channel", "log_subscribers",
        "duration_sec", "log_duration", "age_days", "tags_count"
    ]
    summary_df = merged[cols_to_summarize].describe().T[["count", "mean", "std", "min", "50%", "max"]]
    logger.info("\n" + summary_df.to_string())

    tier_counts = merged.groupby("subscriber_tier")["channel_name"].nunique()
    logger.info(f"Unique Channels by Tier:\n{tier_counts.to_string()}")

    # 6. Export Processed Dataset
    out_csv = DATA_DIR / "processed_payoff_data.csv"
    merged.to_csv(out_csv, index=False)
    logger.info(f"Saved processed payoff dataset: {out_csv} ({len(merged)} rows)")

    # 7. Generate Comprehensive Data Dictionary
    data_dict = [
        {"Variable": "video_id", "Type": "string", "Source": "YouTube Metadata", "Description": "Unique 11-character YouTube video identifier (Primary Key)."},
        {"Variable": "channel_name", "Type": "string", "Source": "YouTube Metadata", "Description": "Creator channel display name."},
        {"Variable": "channel_id", "Type": "string", "Source": "YouTube Metadata", "Description": "Unique YouTube channel identifier (Cluster Identifier, G = 38)."},
        {"Variable": "title", "Type": "string", "Source": "YouTube Metadata", "Description": "Video title text string."},
        {"Variable": "published_at", "Type": "ISO datetime", "Source": "YouTube Metadata", "Description": "Upload timestamp in UTC."},
        {"Variable": "views", "Type": "integer", "Source": "YouTube Metadata", "Description": "Total cumulative video views (strictly positive, min = 898)."},
        {"Variable": "log_views", "Type": "float", "Source": "Constructed", "Description": "Natural log of cumulative views: ln(views). Primary dependent variable."},
        {"Variable": "score_weighted", "Type": "float", "Source": "Click Appeal Component", "Description": "Click Appeal Score X_v = 0.4*X_title + 0.6*X_thumb (in [0, 1])."},
        {"Variable": "x", "Type": "float", "Source": "Constructed", "Description": "Primary own appeal regressor (x = score_weighted)."},
        {"Variable": "x2", "Type": "float", "Source": "Constructed", "Description": "Quadratic own appeal regressor: x^2 (tests concavity condition beta_2 < 0)."},
        {"Variable": "subscribers_raw", "Type": "integer", "Source": "YouTube Metadata", "Description": "Raw channel subscriber count as recorded at scraping time."},
        {"Variable": "subscribers_channel", "Type": "integer", "Source": "Standardized", "Description": "Channel-level median subscriber count S_i (time-invariant creator scale)."},
        {"Variable": "log_subscribers", "Type": "float", "Source": "Constructed", "Description": "Natural log of channel subscribers: ln(S_i). Primary scale control."},
        {"Variable": "duration_sec", "Type": "integer", "Source": "YouTube Metadata", "Description": "Total video length in seconds (min = 13s, max = 18,886s)."},
        {"Variable": "log_duration", "Type": "float", "Source": "Constructed", "Description": "Natural log of duration: ln(duration_sec). Primary content control."},
        {"Variable": "age_days", "Type": "float", "Source": "YouTube Metadata", "Description": "Video age in days since publication to scrape date (min = 30d, max = 759d)."},
        {"Variable": "log_age", "Type": "float", "Source": "Constructed", "Description": "Natural log of age: ln(age_days). Non-linear exposure robustness control."},
        {"Variable": "tags_count", "Type": "integer", "Source": "YouTube Metadata", "Description": "Number of metadata tags attached to video (min = 0, max = 63)."},
        {"Variable": "log_tags_plus1", "Type": "float", "Source": "Constructed", "Description": "Natural log of tag count: ln(tags_count + 1). Robustness check control (Spec R-Tags)."},
        {"Variable": "subscriber_tier", "Type": "categorical", "Source": "Constructed", "Description": "Stratification tier: Tier 1 (<5M), Tier 2 (5M-20M), Tier 3 (>20M)."},
        {"Variable": "category_id", "Type": "integer", "Source": "YouTube Metadata", "Description": "YouTube category ID (e.g. 24 Entertainment, 20 Gaming)."}
    ]
    df_dict = pd.DataFrame(data_dict)
    dict_csv = DATA_DIR / "data_dictionary.csv"
    df_dict.to_csv(dict_csv, index=False)
    logger.info(f"Saved comprehensive data dictionary: {dict_csv}")

    logger.info("=" * 70)
    logger.info("STEP 1 COMPLETED SUCCESSFULLY: DATASET VALIDATED & PREPARED (N = 809)")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_data_preparation()
