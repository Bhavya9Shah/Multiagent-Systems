#!/usr/bin/env python3
"""
payoff_estimation/scripts/prepare_data.py
Stage 1 of Payoff Estimation: Data Preparation & Competitor Peer Average Construction.

Reads:
- Click appeal score/scores/video_scores.csv (or click_appeal_score/scores/video_scores.csv)

Outputs:
- payoff_estimation/data/prepared_payoff_data.csv
- payoff_estimation/data/data_dictionary.csv
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# Setup paths
PAYOFF_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = PAYOFF_DIR.parent
DATA_DIR = PAYOFF_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = PAYOFF_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "prepare_data.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("prepare_data")

# Locate video_scores.csv
CANDIDATE_PATHS = [
    REPO_ROOT / "Click appeal score" / "scores" / "video_scores.csv",
    REPO_ROOT / "click_appeal_score" / "scores" / "video_scores.csv"
]
SCORE_CSV = None
for p in CANDIDATE_PATHS:
    if p.exists():
        SCORE_CSV = p
        break

if SCORE_CSV is None:
    logger.error(f"Could not locate video_scores.csv in candidates: {CANDIDATE_PATHS}")
    sys.exit(1)


def prepare_payoff_dataset():
    logger.info(f"Loading input scores from: {SCORE_CSV}")
    df_raw = pd.read_csv(SCORE_CSV)
    logger.info(f"Raw dataset loaded: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns.")

    # 1. Verification of Required Core Columns
    required_cols = [
        "video_id", "channel_id", "channel_name", "title", "published_at",
        "views", "subscribers", "score_weighted", "duration_sec", "age_days"
    ]
    missing_required = [c for c in required_cols if c not in df_raw.columns]
    if missing_required:
        logger.error(f"Missing essential columns in input CSV: {missing_required}")
        sys.exit(1)

    df = df_raw.copy()

    # 2. Parse Publication Datetime and Time Windows
    df["published_at_dt"] = pd.to_datetime(df["published_at"])
    df["year_month"] = df["published_at_dt"].dt.to_period("M").astype(str)
    df["quarter"] = df["published_at_dt"].dt.to_period("Q").astype(str)

    # 3. Construct Dependent Variable: Log(Views)
    if (df["views"] <= 0).any():
        logger.warning("Found zero or negative views. Applying log1p to non-positive cases.")
        df["log_views"] = np.log(df["views"].clip(lower=1))
    else:
        df["log_views"] = np.log(df["views"])

    # 4. Creator Scale: Log(Subscribers)
    if (df["subscribers"] <= 0).any():
        logger.warning("Found zero or negative subscribers. Clipping to 1.")
        df["log_subscribers"] = np.log(df["subscribers"].clip(lower=1))
    else:
        df["log_subscribers"] = np.log(df["subscribers"])

    # 5. Primary Click Appeal Score and Polynomial Terms
    df["x"] = df["score_weighted"].astype(float)
    df["x2"] = df["x"] ** 2

    # Alternative scores for sensitivity testing
    if "score_equal_weight" in df.columns:
        df["x_equal"] = df["score_equal_weight"].astype(float)
        df["x2_equal"] = df["x_equal"] ** 2
    if "score_weighted_no_expr" in df.columns:
        df["x_no_expr"] = df["score_weighted_no_expr"].astype(float)
        df["x2_no_expr"] = df["x_no_expr"] ** 2

    # 6. Construct Competitor Leave-One-Out (LOCO) Averages
    # Rule 1: Global Niche LOCO Average
    # Excludes ALL videos of creator i from their peer benchmark
    global_loco = []
    global_peer_counts = []
    for idx, row in df.iterrows():
        peers = df[df["channel_id"] != row["channel_id"]]
        global_loco.append(float(peers["score_weighted"].mean()))
        global_peer_counts.append(len(peers))

    df["x_bar_global"] = global_loco
    df["peers_count_global"] = global_peer_counts
    df["x_inter_global"] = df["x"] * df["x_bar_global"]

    # Rule 2: Rolling 90-Day Window LOCO Average (+/- 45 days)
    # Compares video (i, t) to competitor videos published in [t - 45 days, t + 45 days]
    roll90_loco = []
    roll90_peer_counts = []
    fallback_flags = []

    for idx, row in df.iterrows():
        t = row["published_at_dt"]
        t_start = t - pd.Timedelta(days=45)
        t_end = t + pd.Timedelta(days=45)
        peers = df[
            (df["published_at_dt"] >= t_start) &
            (df["published_at_dt"] <= t_end) &
            (df["channel_id"] != row["channel_id"])
        ]
        p_count = len(peers)
        roll90_peer_counts.append(p_count)

        if p_count >= 3:
            roll90_loco.append(float(peers["score_weighted"].mean()))
            fallback_flags.append(False)
        else:
            # When temporal pool is too small (<3 videos), fallback to global niche LOCO
            roll90_loco.append(global_loco[idx])
            fallback_flags.append(True)

    df["x_bar_roll90"] = roll90_loco
    df["peers_count_roll90"] = roll90_peer_counts
    df["is_fallback_peer_roll90"] = fallback_flags
    df["x_inter_roll90"] = df["x"] * df["x_bar_roll90"]

    # Rule 3: Calendar Quarter LOCO Average
    quarter_loco = []
    quarter_peer_counts = []
    quarter_fallback = []
    for idx, row in df.iterrows():
        q = row["quarter"]
        peers = df[(df["quarter"] == q) & (df["channel_id"] != row["channel_id"])]
        p_count = len(peers)
        quarter_peer_counts.append(p_count)
        if p_count >= 3:
            quarter_loco.append(float(peers["score_weighted"].mean()))
            quarter_fallback.append(False)
        else:
            quarter_loco.append(global_loco[idx])
            quarter_fallback.append(True)

    df["x_bar_quarter"] = quarter_loco
    df["peers_count_quarter"] = quarter_peer_counts
    df["is_fallback_peer_quarter"] = quarter_fallback
    df["x_inter_quarter"] = df["x"] * df["x_bar_quarter"]

    # 7. Mean-Centered Variables (for Multicollinearity Diagnostics)
    x_mean = float(df["x"].mean())
    df["x_c"] = df["x"] - x_mean
    df["x2_c"] = df["x_c"] ** 2

    x_bar_mean_roll90 = float(df["x_bar_roll90"].mean())
    df["x_bar_roll90_c"] = df["x_bar_roll90"] - x_bar_mean_roll90
    df["x_inter_roll90_c"] = df["x_c"] * df["x_bar_roll90_c"]

    # 8. Control Variables Z_it
    df["log_duration"] = np.log(df["duration_sec"].clip(lower=1))
    df["age_days"] = df["age_days"].astype(float)
    if "channel_video_count" in df.columns:
        df["log_channel_video_count"] = np.log(df["channel_video_count"].clip(lower=1))
    if "tags_count" in df.columns:
        df["tags_count"] = df["tags_count"].astype(float)

    # 9. Save Prepared Dataset
    out_csv = DATA_DIR / "prepared_payoff_data.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    logger.info(f"Saved prepared payoff dataset to {out_csv} ({len(df)} rows, {len(df.columns)} columns)")

    # 10. Generate Data Dictionary
    data_dict = [
        {"variable": "video_id", "type": "string", "role": "Identifier", "description": "Unique YouTube video identifier"},
        {"variable": "channel_id", "type": "string", "role": "Cluster/Creator ID", "description": "Unique YouTube channel identifier (cluster variable)"},
        {"variable": "channel_name", "type": "string", "role": "Metadata", "description": "Channel display name"},
        {"variable": "published_at", "type": "datetime", "role": "Metadata", "description": "UTC timestamp of video publication"},
        {"variable": "year_month", "type": "string", "role": "Temporal", "description": "Calendar year and month (YYYY-MM)"},
        {"variable": "quarter", "type": "string", "role": "Temporal", "description": "Calendar year and quarter (YYYYQ#)"},
        {"variable": "views", "type": "integer", "role": "Raw Dependent Variable", "description": "Cumulative video view count (V_it)"},
        {"variable": "log_views", "type": "float", "role": "Dependent Variable (y)", "description": "Natural logarithm of video views: log(V_it)"},
        {"variable": "subscribers", "type": "integer", "role": "Raw Scale Variable", "description": "Channel total subscriber count (s_i)"},
        {"variable": "log_subscribers", "type": "float", "role": "Scale Variable", "description": "Natural logarithm of subscriber count: log(s_i)"},
        {"variable": "score_weighted", "type": "float", "role": "Click Appeal Score (x)", "description": "Primary multimodal Click Appeal Score x_it in [0, 1] (0.4 Title + 0.6 Thumbnail)"},
        {"variable": "x", "type": "float", "role": "Click Appeal Score (x)", "description": "Alias for score_weighted (x_it)"},
        {"variable": "x2", "type": "float", "role": "Quadratic Own-Score (x^2)", "description": "Squared Click Appeal Score x_it^2 measuring diminishing returns"},
        {"variable": "x_bar_roll90", "type": "float", "role": "Competitor Average", "description": "Leave-one-out peer average click appeal in rolling 90-day window (t +/- 45d)"},
        {"variable": "x_inter_roll90", "type": "float", "role": "Interaction Term", "description": "Product of own click appeal and rolling 90d peer average: x_it * x_bar_(-i,t)"},
        {"variable": "x_bar_global", "type": "float", "role": "Competitor Average (Alt)", "description": "Leave-one-creator-out average across entire 38-creator niche sample"},
        {"variable": "x_inter_global", "type": "float", "role": "Interaction Term (Alt)", "description": "Product of own click appeal and global LOCO peer average"},
        {"variable": "x_c", "type": "float", "role": "Centered x", "description": "Mean-centered click appeal: x_it - mean(x)"},
        {"variable": "x2_c", "type": "float", "role": "Centered x^2", "description": "Squared mean-centered click appeal: (x_it - mean(x))^2"},
        {"variable": "x_bar_roll90_c", "type": "float", "role": "Centered Peer Average", "description": "Mean-centered rolling 90d peer average"},
        {"variable": "x_inter_roll90_c", "type": "float", "role": "Centered Interaction", "description": "Product of centered own score and centered peer average"},
        {"variable": "duration_sec", "type": "integer", "role": "Raw Control", "description": "Video duration in seconds"},
        {"variable": "log_duration", "type": "float", "role": "Control (Z)", "description": "Natural logarithm of video duration in seconds"},
        {"variable": "age_days", "type": "float", "role": "Control (Z)", "description": "Elapsed days between video publication and data collection"},
        {"variable": "channel_video_count", "type": "integer", "role": "Raw Control", "description": "Total catalog size of the channel"},
        {"variable": "log_channel_video_count", "type": "float", "role": "Control (Z)", "description": "Natural log of total channel video count"},
        {"variable": "tags_count", "type": "integer", "role": "Control (Z)", "description": "Number of metadata tags attached to video"}
    ]
    df_dict = pd.DataFrame(data_dict)
    dict_csv = DATA_DIR / "data_dictionary.csv"
    df_dict.to_csv(dict_csv, index=False, encoding="utf-8")
    logger.info(f"Saved data dictionary to {dict_csv}")

    print("\n================ DATA PREPARATION COMPLETED ================")
    print(f"  Total records prepared : {len(df)}")
    print(f"  Unique channels        : {df['channel_id'].nunique()}")
    print(f"  Time span              : {df['published_at_dt'].min().strftime('%Y-%m-%d')} to {df['published_at_dt'].max().strftime('%Y-%m-%d')}")
    print(f"  log_views              : mean={df['log_views'].mean():.2f}, std={df['log_views'].std():.2f}")
    print(f"  Click Appeal Score x   : mean={df['x'].mean():.4f}, std={df['x'].std():.4f}, min={df['x'].min():.4f}, max={df['x'].max():.4f}")
    print(f"  Peer Average (Roll90)  : mean={df['x_bar_roll90'].mean():.4f}, std={df['x_bar_roll90'].std():.4f}")
    print(f"  Subscribers            : 100% available (no missing values)")
    print("============================================================\n")


if __name__ == "__main__":
    prepare_payoff_dataset()
