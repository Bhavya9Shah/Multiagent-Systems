#!/usr/bin/env python3
"""
payoff_estimation/test_centering_and_peers.py
Examine mean-centering and peer definition variants.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

REPO_ROOT = Path(__file__).resolve().parent.parent
SCORE_CSV = REPO_ROOT / "Click appeal score" / "scores" / "video_scores.csv"

df = pd.read_csv(SCORE_CSV)
df["published_at_dt"] = pd.to_datetime(df["published_at"])
df["quarter"] = df["published_at_dt"].dt.to_period("Q")

df["log_views"] = np.log(df["views"])
df["log_subscribers"] = np.log(df["subscribers"])
df["log_duration"] = np.log(df["duration_sec"])

df["x"] = df["score_weighted"]

# 1. Global LOCO
for idx, r in df.iterrows():
    peers = df[df["channel_id"] != r["channel_id"]]
    df.loc[idx, "x_bar_global"] = peers["score_weighted"].mean()

# 2. Quarterly LOCO
for idx, r in df.iterrows():
    peers = df[(df["quarter"] == r["quarter"]) & (df["channel_id"] != r["channel_id"])]
    if len(peers) >= 3:
        df.loc[idx, "x_bar_quarter"] = peers["score_weighted"].mean()
    else:
        df.loc[idx, "x_bar_quarter"] = df.loc[idx, "x_bar_global"]

# 3. Rolling 90d LOCO
for idx, r in df.iterrows():
    t = r["published_at_dt"]
    peers = df[(df["published_at_dt"] >= t - pd.Timedelta(days=45)) & 
               (df["published_at_dt"] <= t + pd.Timedelta(days=45)) & 
               (df["channel_id"] != r["channel_id"])]
    if len(peers) >= 3:
        df.loc[idx, "x_bar_roll90"] = peers["score_weighted"].mean()
    else:
        df.loc[idx, "x_bar_roll90"] = df.loc[idx, "x_bar_global"]

# Uncentered
df["x2"] = df["x"] ** 2
df["x_inter_roll90"] = df["x"] * df["x_bar_roll90"]
df["x_inter_quarter"] = df["x"] * df["x_bar_quarter"]

# Centered versions (around empirical grand mean of x = 0.3958)
x_mean = df["x"].mean()
df["x_c"] = df["x"] - x_mean
df["x2_c"] = df["x_c"] ** 2
df["x_bar_roll90_c"] = df["x_bar_roll90"] - df["x_bar_roll90"].mean()
df["x_inter_roll90_c"] = df["x_c"] * df["x_bar_roll90_c"]

def print_model_summary(model, cols, name):
    print(f"\n--- {name} ---")
    print(f"R2: {model.rsquared:.4f}, Adj R2: {model.rsquared_adj:.4f}")
    for c in ["const"] + cols:
        print(f"  {c:<20s}: coef={model.params[c]:+9.4f}, se={model.bse[c]:8.4f}, p={model.pvalues[c]:.4f}")
    vifs = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
    print("  VIFs (excluding const):")
    for col_name, v in zip(cols, vifs):
        print(f"    {col_name:<20s}: {v:.2f}")

y = df["log_views"]
groups = df["channel_id"]

# Uncentered Full Model with Rolling 90d
cols_raw = ["x", "x2", "x_bar_roll90", "x_inter_roll90", "log_subscribers", "log_duration", "age_days"]
X_raw = sm.add_constant(df[cols_raw])
m_raw = sm.OLS(y, X_raw).fit(cov_type="cluster", cov_kwds={"groups": groups})
print_model_summary(m_raw, cols_raw, "Raw (Uncentered) Full Model (Rolling 90d)")

# Centered Full Model with Rolling 90d
cols_c = ["x_c", "x2_c", "x_bar_roll90_c", "x_inter_roll90_c", "log_subscribers", "log_duration", "age_days"]
X_c = sm.add_constant(df[cols_c])
m_c = sm.OLS(y, X_c).fit(cov_type="cluster", cov_kwds={"groups": groups})
print_model_summary(m_c, cols_c, "Mean-Centered Full Model (Rolling 90d)")
