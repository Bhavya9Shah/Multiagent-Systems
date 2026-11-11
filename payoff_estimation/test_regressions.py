#!/usr/bin/env python3
"""
payoff_estimation/test_regressions.py
Preliminary regression exploration to test specifications and peer average formulations.
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
df["year_month"] = df["published_at_dt"].dt.to_period("M")

# Dependent variable
df["log_views"] = np.log(df["views"])
df["log_subscribers"] = np.log(df["subscribers"])
df["log_duration"] = np.log(df["duration_sec"])

# Click appeal score
df["x"] = df["score_weighted"]
df["x2"] = df["x"] ** 2

# Definition A: Global LOCO
# For each video, the average click appeal of all videos from other creators
df["x_bar_global"] = 0.0
for idx, r in df.iterrows():
    peers = df[df["channel_id"] != r["channel_id"]]
    df.loc[idx, "x_bar_global"] = peers["score_weighted"].mean()

df["x_inter_global"] = df["x"] * df["x_bar_global"]

# Definition B: Rolling 90-day LOCO (+/- 45 days)
# If a video has < 3 peers in the 90-day window, fallback to global LOCO to prevent erratic outliers
df["x_bar_roll90"] = 0.0
for idx, r in df.iterrows():
    t = r["published_at_dt"]
    peers = df[(df["published_at_dt"] >= t - pd.Timedelta(days=45)) & 
               (df["published_at_dt"] <= t + pd.Timedelta(days=45)) & 
               (df["channel_id"] != r["channel_id"])]
    if len(peers) >= 3:
        df.loc[idx, "x_bar_roll90"] = peers["score_weighted"].mean()
    else:
        # fallback to global LOCO
        df.loc[idx, "x_bar_roll90"] = df.loc[idx, "x_bar_global"]

df["x_inter_roll90"] = df["x"] * df["x_bar_roll90"]

print("x_bar_global range:", df["x_bar_global"].min(), df["x_bar_global"].max(), "std:", df["x_bar_global"].std())
print("x_bar_roll90 range:", df["x_bar_roll90"].min(), df["x_bar_roll90"].max(), "std:", df["x_bar_roll90"].std())

def run_ols(formula_desc, X_cols, peer_desc=""):
    X = df[X_cols]
    X = sm.add_constant(X)
    y = df["log_views"]
    
    # OLS with Cluster-Robust standard errors (clustered on channel_id)
    model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df["channel_id"]})
    
    print(f"\n=======================================================")
    print(f"MODEL: {formula_desc} ({peer_desc})")
    print(f"R-squared: {model.rsquared:.4f}, Adj R-squared: {model.rsquared_adj:.4f}, N: {model.nobs:.0f}, Clusters: {df['channel_id'].nunique()}")
    print("Coefficients and p-values (Cluster-Robust SE):")
    res_df = pd.DataFrame({
        "coef": model.params,
        "std err": model.bse,
        "t / z": model.tvalues,
        "P>|z|": model.pvalues,
        "[0.025": model.conf_int()[0],
        "0.975]": model.conf_int()[1]
    })
    print(res_df.to_string())
    
    # VIF
    if len(X_cols) > 1:
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_cols
        vif_data["VIF"] = [variance_inflation_factor(X[X_cols].values, i) for i in range(len(X_cols))]
        print("\nVariance Inflation Factors (VIF):")
        print(vif_data.to_string(index=False))
        
    return model

# 1. Model 1: Baseline Linear (x, log_subscribers, log_duration, age_days)
run_ols("M1: Baseline Linear", ["x", "log_subscribers", "log_duration", "age_days"])

# 2. Model 2: Quadratic own-score (x, x2, log_subscribers, log_duration, age_days)
run_ols("M2: Quadratic Own-Score", ["x", "x2", "log_subscribers", "log_duration", "age_days"])

# 3. Model 3A: Full Model with Global LOCO
run_ols("M3A: Full Model (Global LOCO)", ["x", "x2", "x_bar_global", "x_inter_global", "log_subscribers", "log_duration", "age_days"], "Global LOCO")

# 4. Model 3B: Full Model with Rolling 90-day LOCO
run_ols("M3B: Full Model (Rolling 90d LOCO)", ["x", "x2", "x_bar_roll90", "x_inter_roll90", "log_subscribers", "log_duration", "age_days"], "Rolling 90d LOCO")
