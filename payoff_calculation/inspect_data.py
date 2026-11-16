#!/usr/bin/env python3
"""
payoff_estimation/inspect_data.py
Exploratory data analysis and feasibility report for empirical payoff estimation.
"""

from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SCORE_CSV = REPO_ROOT / "Click appeal score" / "scores" / "video_scores.csv"

if not SCORE_CSV.exists():
    # Also check lowercase click_appeal_score
    alt = REPO_ROOT / "click_appeal_score" / "scores" / "video_scores.csv"
    if alt.exists():
        SCORE_CSV = alt
    else:
        raise FileNotFoundError(f"Cannot find video_scores.csv at {SCORE_CSV}")

df = pd.read_csv(SCORE_CSV)
print("Loaded video_scores.csv from:", SCORE_CSV)
print("Rows:", len(df), "Columns:", len(df.columns))

# Parse dates
df["published_at_dt"] = pd.to_datetime(df["published_at"])
df["year_month"] = df["published_at_dt"].dt.to_period("M")
df["quarter"] = df["published_at_dt"].dt.to_period("Q")

# 1. Creators
print("\n--- CREATOR STRUCTURE ---")
channel_counts = df["channel_name"].value_counts()
print(f"Total Unique Channels: {df['channel_name'].nunique()}")
print(f"Channels with 5 videos: {(channel_counts == 5).sum()}")
print(f"Channels with != 5 videos: {(channel_counts != 5).sum()} -> {channel_counts[channel_counts != 5].to_dict()}")

# 2. Time coverage
print("\n--- TIME COVERAGE ---")
print(f"Earliest: {df['published_at_dt'].min()}")
print(f"Latest:   {df['published_at_dt'].max()}")
print(f"Span:     {(df['published_at_dt'].max() - df['published_at_dt'].min()).days} days")
print(f"Total Calendar Months: {df['year_month'].nunique()}")
print(f"Total Calendar Quarters: {df['quarter'].nunique()}")

# 3. Subscriber Counts
print("\n--- SUBSCRIBERS ---")
print("Null count:", df["subscribers"].isnull().sum())
print("Summary stats:")
print(df["subscribers"].describe())

# 4. Views & log(views)
print("\n--- VIEWS & LOG(VIEWS) ---")
print("Null views:", df["views"].isnull().sum())
print("Zero views:", (df["views"] <= 0).sum())
print("Views summary:")
print(df["views"].describe())
df["log_views"] = np.log(df["views"])
print("log(views) summary:")
print(df["log_views"].describe())

# 5. Peer average feasibility
print("\n--- PEER AVERAGE CANDIDATES ---")
# Definition 1: Monthly LOCO
monthly_peer_counts = []
for idx, r in df.iterrows():
    peers = df[(df["year_month"] == r["year_month"]) & (df["channel_id"] != r["channel_id"])]
    monthly_peer_counts.append(len(peers))
df["monthly_peers_count"] = monthly_peer_counts

print("Monthly peers per video:")
print(pd.Series(monthly_peer_counts).describe())
zero_monthly = (pd.Series(monthly_peer_counts) == 0).sum()
small_monthly = (pd.Series(monthly_peer_counts) < 5).sum()
print(f"Videos with 0 peers in same month: {zero_monthly}")
print(f"Videos with < 5 peers in same month: {small_monthly} ({small_monthly/len(df)*100:.1f}%)")

# Definition 2: Rolling 60-day window (+/- 30 days)
roll60_peer_counts = []
for idx, r in df.iterrows():
    t = r["published_at_dt"]
    peers = df[(df["published_at_dt"] >= t - pd.Timedelta(days=30)) & 
               (df["published_at_dt"] <= t + pd.Timedelta(days=30)) & 
               (df["channel_id"] != r["channel_id"])]
    roll60_peer_counts.append(len(peers))
print("\nRolling 60-day (+/- 30d) peers per video:")
print(pd.Series(roll60_peer_counts).describe())
print(f"Videos with < 5 peers in 60d window: {(pd.Series(roll60_peer_counts) < 5).sum()}")

# Definition 3: Rolling 90-day window (+/- 45 days)
roll90_peer_counts = []
for idx, r in df.iterrows():
    t = r["published_at_dt"]
    peers = df[(df["published_at_dt"] >= t - pd.Timedelta(days=45)) & 
               (df["published_at_dt"] <= t + pd.Timedelta(days=45)) & 
               (df["channel_id"] != r["channel_id"])]
    roll90_peer_counts.append(len(peers))
print("\nRolling 90-day (+/- 45d) peers per video:")
print(pd.Series(roll90_peer_counts).describe())
print(f"Videos with < 5 peers in 90d window: {(pd.Series(roll90_peer_counts) < 5).sum()}")

# Definition 4: Global Niche LOCO (Leave-One-Creator-Out)
global_peers_count = []
for idx, r in df.iterrows():
    peers = df[df["channel_id"] != r["channel_id"]]
    global_peers_count.append(len(peers))
print("\nGlobal Niche LOCO peers per video:")
print(pd.Series(global_peers_count).describe())

# 6. Controls
print("\n--- CANDIDATE CONTROLS ---")
controls = ["duration_sec", "age_days", "tags_count", "category_id", "channel_video_count"]
for c in controls:
    print(f"{c:20s}: min={df[c].min():8.1f}, max={df[c].max():8.1f}, mean={df[c].mean():8.2f}, nulls={df[c].isnull().sum()}")

# 7. Correlation matrix of core variables
print("\n--- CORRELATIONS WITH LOG(VIEWS) ---")
core_cols = ["log_views", "score_weighted", "subscribers", "duration_sec", "age_days", "tags_count", "channel_video_count"]
df["log_subscribers"] = np.log(df["subscribers"])
df["log_duration"] = np.log(df["duration_sec"])
corr_cols = ["log_views", "score_weighted", "log_subscribers", "log_duration", "age_days", "tags_count", "channel_video_count"]
corr = df[corr_cols].corr()
print(corr["log_views"].to_string())
