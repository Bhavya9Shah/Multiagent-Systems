#!/usr/bin/env python3
"""
payoff_estimation/scripts/03_train_models.py
Phase 2 - Step 3: Primary Model Development on 26 Training Channels.

Requirements:
1. Training sample: Exactly 26 channels, 541 videos.
2. Peer construction: Contemporaneous window [t - 30d, t + 30d], strictly within the other 25 training channels.
3. Model estimation:
   - Nested Models: M0, M1, M2, M3, M4, M4-Centered.
   - Standard Errors: OLS homoskedastic, HC3 robust, and Channel-Clustered (CRSE, G = 26).
4. Robustness specifications:
   - Spec A: Primary M4-Centered with log(subscribers).
   - Spec B: Omitted-scale sensitivity (no subscribers).
   - Spec C: Channel Fixed Effects (within-channel explanatory analysis).
   - Spec R-Tags: Additional log(tags + 1) control.
5. In-sample Information Criteria: AIC and BIC computed strictly on training estimation sample.
6. Diagnostics: VIF, residual normality (Jarque-Bera), heteroskedasticity (Breusch-Pagan).
7. Exports:
   - payoff_estimation/results/training_regression_results.csv
   - payoff_estimation/results/training_model_comparison.csv
   - payoff_estimation/results/subscriber_robustness_table.csv
   - payoff_estimation/results/training_multicollinearity_diagnostics.csv
   - payoff_estimation/logs/03_train_models.log
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PAYOFF_DIR = SCRIPT_DIR.parent
DATA_DIR = PAYOFF_DIR / "data"
RESULTS_DIR = PAYOFF_DIR / "results"
LOGS_DIR = PAYOFF_DIR / "logs"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Logging
LOG_FILE = LOGS_DIR / "03_train_models.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("03_train_models")


def compute_training_peers(train_df):
    """
    Construct channel-weighted contemporaneous peer average strictly among the other 25 training channels.
    Window: [t - 30 days, t + 30 days].
    """
    logger.info("Constructing contemporaneous peer averages strictly among eligible training channels...")
    train_df = train_df.copy()
    train_df["published_at_dt"] = pd.to_datetime(train_df["published_at"])
    
    peer_averages = []
    active_channel_counts = []
    peer_video_counts = []
    
    for idx, row in train_df.iterrows():
        c_id = row["channel_id"]
        t_pub = row["published_at_dt"]
        
        # Filter: competitor channels within training split and inside [t-30d, t+30d]
        window_df = train_df[
            (train_df["channel_id"] != c_id) &
            (train_df["published_at_dt"] >= t_pub - pd.Timedelta(days=30)) &
            (train_df["published_at_dt"] <= t_pub + pd.Timedelta(days=30))
        ]
        
        if len(window_df) > 0:
            # Channel-weighted average: mean appeal per active channel, then average across channels
            ch_means = window_df.groupby("channel_id")["x"].mean()
            peer_averages.append(float(ch_means.mean()))
            active_channel_counts.append(len(ch_means))
            peer_video_counts.append(len(window_df))
        else:
            # Fallback: full training LOCO for channel i
            loco_df = train_df[train_df["channel_id"] != c_id]
            ch_means = loco_df.groupby("channel_id")["x"].mean()
            peer_averages.append(float(ch_means.mean()))
            active_channel_counts.append(len(ch_means))
            peer_video_counts.append(len(loco_df))
            
    train_df["x_bar"] = peer_averages
    train_df["x_inter"] = train_df["x"] * train_df["x_bar"]
    train_df["peer_active_channels"] = active_channel_counts
    train_df["peer_video_count"] = peer_video_counts
    
    logger.info(f"Training peer appeal constructed: mean = {train_df['x_bar'].mean():.4f}, std = {train_df['x_bar'].std():.4f}")
    logger.info(f"Active competitor channels per window: min = {min(active_channel_counts)}, median = {np.median(active_channel_counts)}, max = {max(active_channel_counts)}")
    return train_df


def run_training_estimations():
    logger.info("=" * 70)
    logger.info("STEP 3: PRIMARY MODEL DEVELOPMENT ON 26 TRAINING CHANNELS")
    logger.info("=" * 70)
    
    input_csv = DATA_DIR / "partitioned_payoff_data.csv"
    if not input_csv.exists():
        logger.error(f"Cannot find partitioned dataset: {input_csv}")
        sys.exit(1)
        
    full_df = pd.read_csv(input_csv)
    train_df = full_df[full_df["partition"] == "train"].copy()
    
    N_train = len(train_df)
    G_train = train_df["channel_id"].nunique()
    logger.info(f"Isolated Training Sample: N = {N_train} videos across G = {G_train} channels.")
    assert N_train == 541, f"Expected 541 training videos, got {N_train}"
    assert G_train == 26, f"Expected 26 training channels, got {G_train}"
    
    # 1. Peer Variable Construction
    train_df = compute_training_peers(train_df)
    
    # 2. Centered Regressors (Estimated strictly from training sample means)
    mu_x = float(train_df["x"].mean())
    mu_xbar = float(train_df["x_bar"].mean())
    logger.info(f"Training fold centering parameters: mu_x = {mu_x:.6f}, mu_xbar = {mu_xbar:.6f}")
    
    train_df["x_c"] = train_df["x"] - mu_x
    train_df["x2_c"] = train_df["x_c"] ** 2
    train_df["x_bar_c"] = train_df["x_bar"] - mu_xbar
    train_df["x_inter_c"] = train_df["x_c"] * train_df["x_bar_c"]
    train_df["const"] = 1.0
    
    # 3. Model Specifications Dictionary
    base_controls = ["log_subscribers", "log_duration", "age_days"]
    
    nested_specs = {
        "Model_0_Baseline": ["const"] + base_controls,
        "Model_1_Linear": ["const", "x"] + base_controls,
        "Model_2_Quadratic": ["const", "x", "x2"] + base_controls,
        "Model_3_PeerCongestion": ["const", "x", "x2", "x_bar"] + base_controls,
        "Model_4_Interaction": ["const", "x", "x2", "x_bar", "x_inter"] + base_controls,
        "Model_4_Centered": ["const", "x_c", "x2_c", "x_bar_c", "x_inter_c"] + base_controls
    }
    
    # 4. Estimation Loop for Nested Models
    reg_rows = []
    comp_rows = []
    vif_rows = []
    
    y = train_df["log_views"].values
    cluster_groups = train_df["channel_id"].values
    
    for model_name, var_list in nested_specs.items():
        X = train_df[var_list]
        K = X.shape[1]
        
        # Fit with clustered standard errors
        fit_crse = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": cluster_groups})
        # Fit with HC3 robust standard errors
        fit_hc3 = sm.OLS(y, X).fit(cov_type="HC3")
        # Fit with classical OLS standard errors
        fit_ols = sm.OLS(y, X).fit()
        
        # In-sample Information Criteria
        ll = float(fit_ols.llf)
        aic = float(fit_ols.aic)
        bic = float(fit_ols.bic)
        r2 = float(fit_ols.rsquared)
        adj_r2 = float(fit_ols.rsquared_adj)
        ssr = float(fit_ols.ssr)
        
        # Diagnostics
        resid = fit_ols.resid
        jb_stat, jb_p = stats.jarque_bera(resid)
        bp_stat, bp_p, _, _ = het_breuschpagan(resid, X)
        
        comp_rows.append({
            "model_name": model_name,
            "n_obs": N_train,
            "n_clusters": G_train,
            "k_params": K,
            "r_squared": r2,
            "adj_r_squared": adj_r2,
            "ssr": ssr,
            "log_likelihood": ll,
            "aic": aic,
            "bic": bic,
            "jb_stat": jb_stat,
            "jb_p_value": jb_p,
            "bp_stat": bp_stat,
            "bp_p_value": bp_p
        })
        
        # Variable-level estimates
        for i, var in enumerate(var_list):
            coef = float(fit_crse.params[var])
            se_crse = float(fit_crse.bse[var])
            se_hc3 = float(fit_hc3.bse[var])
            se_ols = float(fit_ols.bse[var])
            t_stat = float(fit_crse.tvalues[var])
            p_val = float(fit_crse.pvalues[var])
            
            # 95% CI based on CRSE
            crit_t = stats.t.ppf(0.975, df=G_train - 1)
            ci_lower = coef - crit_t * se_crse
            ci_upper = coef + crit_t * se_crse
            
            # VIF
            if var != "const" and K > 2:
                idx_vif = list(X.columns).index(var)
                vif = float(variance_inflation_factor(X.values, idx_vif))
            else:
                vif = np.nan
                
            reg_rows.append({
                "model_name": model_name,
                "variable": var,
                "coef": coef,
                "se_crse": se_crse,
                "se_hc3": se_hc3,
                "se_ols": se_ols,
                "t_stat_crse": t_stat,
                "p_val_crse": p_val,
                "ci_lower_95": ci_lower,
                "ci_upper_95": ci_upper,
                "vif": vif,
                "sample_size": N_train,
                "cluster_count": G_train
            })
            
            if var != "const":
                vif_rows.append({
                    "model_name": model_name,
                    "variable": var,
                    "vif": vif
                })
                
        logger.info(f"{model_name:<25}: R² = {r2:.4f}, Adj R² = {adj_r2:.4f}, AIC = {aic:.1f}, BIC = {bic:.1f}")

    # 5. Robustness Specifications (Specs A, B, C, R-Tags)
    robustness_specs = {}
    
    # Spec A: Primary M4-Centered (already estimated, reference)
    robustness_specs["Spec_A_Primary"] = nested_specs["Model_4_Centered"]
    
    # Spec B: Omitted-Scale Sensitivity (omits log_subscribers)
    robustness_specs["Spec_B_OmittedScale"] = ["const", "x_c", "x2_c", "x_bar_c", "x_inter_c", "log_duration", "age_days"]
    
    # Spec C: Channel Fixed Effects (within-channel explanatory analysis)
    fe_dummies = pd.get_dummies(train_df["channel_id"], drop_first=True, dtype=float)
    fe_dummy_cols = [f"fe_{c}" for c in fe_dummies.columns]
    fe_dummies.columns = fe_dummy_cols
    train_fe_df = pd.concat([train_df, fe_dummies], axis=1)
    robustness_specs["Spec_C_ChannelFE"] = ["const", "x_c", "x2_c", "x_bar_c", "x_inter_c", "log_duration", "age_days"] + fe_dummy_cols
    
    # Spec R-Tags: Additional log(tags + 1) control
    robustness_specs["Spec_R_Tags"] = nested_specs["Model_4_Centered"] + ["log_tags_plus1"]
    
    robustness_rows = []
    
    for r_name, r_vars in robustness_specs.items():
        if r_name == "Spec_C_ChannelFE":
            X_r = train_fe_df[r_vars]
        else:
            X_r = train_df[r_vars]
            
        fit_r = sm.OLS(y, X_r).fit(cov_type="cluster", cov_kwds={"groups": cluster_groups})
        fit_r_ols = sm.OLS(y, X_r).fit()
        
        core_vars = ["x_c", "x2_c", "x_bar_c", "x_inter_c"]
        for cv in core_vars:
            coef = float(fit_r.params[cv])
            se_crse = float(fit_r.bse[cv])
            p_val = float(fit_r.pvalues[cv])
            crit_t = stats.t.ppf(0.975, df=G_train - 1)
            
            robustness_rows.append({
                "specification": r_name,
                "variable": cv,
                "coef": coef,
                "se_crse": se_crse,
                "t_stat": coef / se_crse,
                "p_value": p_val,
                "ci_lower_95": coef - crit_t * se_crse,
                "ci_upper_95": coef + crit_t * se_crse,
                "r_squared": float(fit_r_ols.rsquared),
                "adj_r_squared": float(fit_r_ols.rsquared_adj),
                "aic": float(fit_r_ols.aic),
                "bic": float(fit_r_ols.bic),
                "n_obs": N_train,
                "n_clusters": G_train
            })
            
        logger.info(f"Robustness {r_name:<25}: R² = {fit_r_ols.rsquared:.4f}, AIC = {fit_r_ols.aic:.1f}, BIC = {fit_r_ols.bic:.1f}")

    # 6. Export Tables
    df_reg = pd.DataFrame(reg_rows)
    df_comp = pd.DataFrame(comp_rows)
    df_rob = pd.DataFrame(robustness_rows)
    df_vif = pd.DataFrame(vif_rows)
    
    out_reg = RESULTS_DIR / "training_regression_results.csv"
    out_comp = RESULTS_DIR / "training_model_comparison.csv"
    out_rob = RESULTS_DIR / "subscriber_robustness_table.csv"
    out_vif = RESULTS_DIR / "training_multicollinearity_diagnostics.csv"
    
    df_reg.to_csv(out_reg, index=False)
    df_comp.to_csv(out_comp, index=False)
    df_rob.to_csv(out_rob, index=False)
    df_vif.to_csv(out_vif, index=False)
    
    logger.info(f"Saved training regression results: {out_reg}")
    logger.info(f"Saved model comparison table: {out_comp}")
    logger.info(f"Saved robustness table: {out_rob}")
    logger.info(f"Saved multicollinearity diagnostics: {out_vif}")
    
    # Also save training peer features to data/train_prepared_data.csv for subsequent steps
    train_prepared_path = DATA_DIR / "train_prepared_data.csv"
    train_df.to_csv(train_prepared_path, index=False)
    logger.info(f"Saved prepared training dataset with peer features: {train_prepared_path}")
    
    logger.info("=" * 70)
    logger.info("STEP 3 COMPLETED SUCCESSFULLY: MODELS ESTIMATED ON 26 TRAINING CHANNELS")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_training_estimations()
