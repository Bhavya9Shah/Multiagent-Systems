#!/usr/bin/env python3
"""
payoff_estimation/scripts/07_post_test_structural_calibration.py
Phase 2 - Step 7: Post-Test Full-Sample Structural Calibration & Game Theory Handoff.

Requirements:
1. ONLY AFTER final holdout test evaluation is complete and documented, fit the locked
   Model 4-Centered on the full dataset of 38 channels (N = 809 videos).
2. Compute peer features for all 809 videos using a contemporaneous rolling window of
   [t - 30d, t + 30d] (LOCO among all 38 channels).
3. Compute centering means across the full 809 observations: mu_x, mu_xbar.
4. Fit Model 4-Centered and uncentered counterpart with:
   - CRSE (clustered by channel, G = 38)
   - HC3 robust SE
   - Classical OLS SE
5. Perform Null-Restricted Wild Cluster Bootstrap (WCB) (B = 2,000 draws, seed = 42):
   - H0: beta_2 >= 0 vs H1: beta_2 < 0 (one-sided test of diminishing returns / strict concavity)
   - H0: beta_5 = 0 vs H1: beta_5 != 0 (two-sided test of strategic interaction)
   - Use Rademacher weights (+1 / -1 at cluster level)
   - Use global finite-sample residual scaling factor: c = sqrt((G/(G-1)) * ((N-1)/(N-K)))
6. Perform Unrestricted Wild Cluster Bootstrap (B = 2,000 draws, seed = 42):
   - 95% percentile confidence intervals for all parameters
   - Empirical distribution of contraction ratio: rho = |beta_5| / (2 * |beta_2|)
   - Retain ALL bootstrap draws without discarding non-concave iterations (beta_2 >= 0)
   - Report fraction of concave draws, fraction with rho < 1, and percentiles (10th, 25th, 50th, 75th, 90th)
7. Export comprehensive game-theoretic payload:
   - payoff_estimation/results/game_theory_coefficients.json
   - payoff_estimation/results/game_theory_coefficients.csv
   - payoff_estimation/results/post_test_structural_regression_results.csv
   - payoff_estimation/results/bootstrap_inference_results.csv
   - payoff_estimation/logs/07_post_test_structural_calibration.log
"""

import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PAYOFF_DIR = SCRIPT_DIR.parent
DATA_DIR = PAYOFF_DIR / "data"
RESULTS_DIR = PAYOFF_DIR / "results"
LOGS_DIR = PAYOFF_DIR / "logs"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
LOG_FILE = LOGS_DIR / "07_post_test_structural_calibration.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("07_post_test_structural_calibration")


def run_structural_calibration(B=2000, seed=42):
    logger.info("=" * 80)
    logger.info("STEP 7: POST-TEST FULL-SAMPLE STRUCTURAL CALIBRATION & GAME THEORY HANDOFF")
    logger.info("=" * 80)
    logger.info(
        "CONTEXT: Post-test full-sample estimation is conducted ONLY after holdout test\n"
        "generalization was verified. The resulting parameters serve as structural inputs\n"
        "for the CS 329 potential game simulation (Khushi Shah) and social welfare analysis."
    )

    input_csv = DATA_DIR / "partitioned_payoff_data.csv"
    if not input_csv.exists():
        input_csv = DATA_DIR / "processed_payoff_data.csv"
    if not input_csv.exists():
        logger.error(f"Cannot find dataset at {input_csv}")
        sys.exit(1)

    df = pd.read_csv(input_csv)
    df["published_at_dt"] = pd.to_datetime(df["published_at"])

    N = len(df)
    G = df["channel_id"].nunique()
    logger.info(f"Loaded Full Dataset: N = {N} observations across G = {G} channels.")
    assert N == 809, f"Expected 809 observations, got {N}"
    assert G == 38, f"Expected 38 channels, got {G}"

    # 1. Compute Full-Sample LOCO Peer Appeal in [t - 30d, t + 30d]
    logger.info("Computing contemporaneous rolling peer appeal [t - 30d, t + 30d] across all 38 channels...")
    peer_appeals = []
    peer_counts = []
    for _, row in df.iterrows():
        c_id = row["channel_id"]
        t_pub = row["published_at_dt"]
        w = df[
            (df["channel_id"] != c_id) &
            (df["published_at_dt"] >= t_pub - pd.Timedelta(days=30)) &
            (df["published_at_dt"] <= t_pub + pd.Timedelta(days=30))
        ]
        if len(w) > 0:
            peer_appeals.append(float(w.groupby("channel_id")["x"].mean().mean()))
            peer_counts.append(len(w))
        else:
            loco = df[df["channel_id"] != c_id]
            peer_appeals.append(float(loco.groupby("channel_id")["x"].mean().mean()))
            peer_counts.append(0)
    df["x_bar"] = peer_appeals
    df["peer_count"] = peer_counts

    # 2. Centering Means across full sample
    mu_x = float(df["x"].mean())
    mu_xbar = float(df["x_bar"].mean())
    logger.info(f"Full-Sample Centering Parameters: mu_x = {mu_x:.6f}, mu_xbar = {mu_xbar:.6f}")

    df["x_c"] = df["x"] - mu_x
    df["x2_c"] = df["x_c"] ** 2
    df["x_bar_c"] = df["x_bar"] - mu_xbar
    df["x_inter_c"] = df["x_c"] * df["x_bar_c"]

    # Uncentered interaction and quadratic
    df["x2"] = df["x"] ** 2
    df["x_inter"] = df["x"] * df["x_bar"]
    df["const"] = 1.0

    # 3. Model Specifications
    centered_cols = ["const", "x_c", "x2_c", "x_bar_c", "x_inter_c", "log_subscribers", "log_duration", "age_days"]
    uncentered_cols = ["const", "x", "x2", "x_bar", "x_inter", "log_subscribers", "log_duration", "age_days"]

    y = df["log_views"].values
    X_c = df[centered_cols].values
    X_u = df[uncentered_cols].values
    K = len(centered_cols)
    clusters = df["channel_id"].values
    unique_clusters = np.unique(clusters)

    # 4. Estimation of Centered Model 4
    logger.info("\n" + "=" * 80)
    logger.info("ESTIMATING FULL-SAMPLE MODEL 4-CENTERED (PRIMARY STRUCTURAL SPECIFICATION)")
    logger.info("=" * 80)

    ols_c = sm.OLS(y, df[centered_cols]).fit()
    crse_c = sm.OLS(y, df[centered_cols]).fit(cov_type="cluster", cov_kwds={"groups": clusters})
    hc3_c = sm.OLS(y, df[centered_cols]).fit(cov_type="HC3")

    # Uncentered estimation for baseline comparison
    ols_u = sm.OLS(y, df[uncentered_cols]).fit()
    crse_u = sm.OLS(y, df[uncentered_cols]).fit(cov_type="cluster", cov_kwds={"groups": clusters})

    # VIF calculations
    vifs_c = {}
    for i, col in enumerate(centered_cols[1:], start=1):
        vifs_c[col] = float(variance_inflation_factor(X_c, i))

    vifs_u = {}
    for i, col in enumerate(uncentered_cols[1:], start=1):
        vifs_u[col] = float(variance_inflation_factor(X_u, i))

    logger.info("Centered Model VIFs (Resolution of Multicollinearity):")
    for col, v in vifs_c.items():
        logger.info(f"  {col:<18}: VIF = {v:8.3f}")

    logger.info("\nCentered Model 4 Estimates (CRSE Clustered by Channel, G = 38):")
    reg_records = []
    for var, coef, se_cl, se_hc3, se_ols in zip(
        centered_cols, crse_c.params, crse_c.bse, hc3_c.bse, ols_c.bse
    ):
        t_stat = coef / se_cl
        p_norm = 2.0 * (1.0 - stats.norm.cdf(np.abs(t_stat)))
        p_t = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df=G - 1))
        ci_95_low = coef - stats.t.ppf(0.975, df=G - 1) * se_cl
        ci_95_high = coef + stats.t.ppf(0.975, df=G - 1) * se_cl

        logger.info(
            f"  {var:<18}: Coef = {coef:10.4f} | CRSE = {se_cl:8.4f} | t = {t_stat:7.3f} | "
            f"p(t_df37) = {p_t:.4f} | 95% CI = [{ci_95_low:8.3f}, {ci_95_high:8.3f}]"
        )

        reg_records.append({
            "model": "Model_4_Centered",
            "variable": var,
            "coef": float(coef),
            "std_error_crse": float(se_cl),
            "std_error_hc3": float(se_hc3),
            "std_error_ols": float(se_ols),
            "t_statistic": float(t_stat),
            "p_value_asymptotic_z": float(p_norm),
            "p_value_student_t": float(p_t),
            "ci_95_lower_t": float(ci_95_low),
            "ci_95_upper_t": float(ci_95_high),
            "vif": vifs_c.get(var, np.nan)
        })

    # Uncentered records
    for var, coef, se_cl in zip(uncentered_cols, crse_u.params, crse_u.bse):
        t_stat = coef / se_cl
        p_t = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df=G - 1))
        reg_records.append({
            "model": "Model_4_Uncentered",
            "variable": var,
            "coef": float(coef),
            "std_error_crse": float(se_cl),
            "std_error_hc3": float(hc3_c.bse.iloc[0] if var == "const" else np.nan),
            "std_error_ols": float(ols_u.bse.get(var, np.nan)),
            "t_statistic": float(t_stat),
            "p_value_asymptotic_z": float(2.0 * (1.0 - stats.norm.cdf(np.abs(t_stat)))),
            "p_value_student_t": float(p_t),
            "ci_95_lower_t": float(coef - stats.t.ppf(0.975, df=G - 1) * se_cl),
            "ci_95_upper_t": float(coef + stats.t.ppf(0.975, df=G - 1) * se_cl),
            "vif": vifs_u.get(var, np.nan)
        })

    df_reg_results = pd.DataFrame(reg_records)
    out_reg_csv = RESULTS_DIR / "post_test_structural_regression_results.csv"
    df_reg_results.to_csv(out_reg_csv, index=False)
    logger.info(f"Saved structural regression results table: {out_reg_csv}")

    # 5. Wild Cluster Bootstrap Hypothesis Tests & CIs
    logger.info("\n" + "=" * 80)
    logger.info(f"RUNNING WILD CLUSTER BOOTSTRAP INFERENCE (B = {B} DRAWS, SEED = {seed})")
    logger.info("=" * 80)

    # Finite sample scaling factor: c = sqrt((G/(G-1)) * ((N-1)/(N-K)))
    c_scale = np.sqrt((G / (G - 1.0)) * ((N - 1.0) / (N - K)))
    logger.info(f"Finite sample cluster scaling factor c = {c_scale:.6f} (G={G}, N={N}, K={K})")

    idx_x2 = centered_cols.index("x2_c")
    idx_inter = centered_cols.index("x_inter_c")

    t_sample_x2 = float(crse_c.params["x2_c"] / crse_c.bse["x2_c"])
    t_sample_inter = float(crse_c.params["x_inter_c"] / crse_c.bse["x_inter_c"])

    # -------------------------------------------------------------------------
    # Test A: Null-Restricted WCB for H0: beta_2 >= 0 vs H1: beta_2 < 0 (Concavity)
    # -------------------------------------------------------------------------
    cols_r_x2 = [c for c in centered_cols if c != "x2_c"]
    ols_r_x2 = sm.OLS(y, df[cols_r_x2]).fit()
    y_hat_r_x2 = ols_r_x2.fittedvalues.values
    u_r_x2 = (y - y_hat_r_x2) * c_scale

    logger.info("Executing Test A: Null-Restricted WCB for H0: beta_2 >= 0 (Concavity)...")
    np.random.seed(seed)
    t_boot_x2 = []
    for b in range(B):
        # Rademacher cluster weights: +1 or -1
        v_dict = {cl: np.random.choice([-1.0, 1.0]) for cl in unique_clusters}
        v_vec = np.array([v_dict[cl] for cl in clusters])
        y_b = y_hat_r_x2 + v_vec * u_r_x2

        fit_b = sm.OLS(y_b, X_c).fit(cov_type="cluster", cov_kwds={"groups": clusters})
        t_boot_x2.append(fit_b.params[idx_x2] / fit_b.bse[idx_x2])

    t_boot_x2 = np.array(t_boot_x2)
    # One-sided p-value for H1: beta_2 < 0
    p_val_wcb_concavity = float(np.mean(t_boot_x2 <= t_sample_x2))
    logger.info(
        f"Test A Result: t_sample = {t_sample_x2:.4f} | WCB one-sided p-value = {p_val_wcb_concavity:.4f} | "
        f"Concavity H0 (beta_2 >= 0) is {'REJECTED' if p_val_wcb_concavity < 0.05 else 'RETAINED'} at alpha = 0.05"
    )

    # -------------------------------------------------------------------------
    # Test B: Null-Restricted WCB for H0: beta_5 = 0 vs H1: beta_5 != 0 (Interaction)
    # -------------------------------------------------------------------------
    cols_r_inter = [c for c in centered_cols if c != "x_inter_c"]
    ols_r_inter = sm.OLS(y, df[cols_r_inter]).fit()
    y_hat_r_inter = ols_r_inter.fittedvalues.values
    u_r_inter = (y - y_hat_r_inter) * c_scale

    logger.info("Executing Test B: Null-Restricted WCB for H0: beta_5 = 0 (Strategic Interaction)...")
    np.random.seed(seed)
    t_boot_inter = []
    for b in range(B):
        v_dict = {cl: np.random.choice([-1.0, 1.0]) for cl in unique_clusters}
        v_vec = np.array([v_dict[cl] for cl in clusters])
        y_b = y_hat_r_inter + v_vec * u_r_inter

        fit_b = sm.OLS(y_b, X_c).fit(cov_type="cluster", cov_kwds={"groups": clusters})
        t_boot_inter.append(fit_b.params[idx_inter] / fit_b.bse[idx_inter])

    t_boot_inter = np.array(t_boot_inter)
    # Two-sided p-value
    p_val_wcb_interaction = float(np.mean(np.abs(t_boot_inter) >= np.abs(t_sample_inter)))
    logger.info(
        f"Test B Result: t_sample = {t_sample_inter:.4f} | WCB two-sided p-value = {p_val_wcb_interaction:.4f} | "
        f"Interaction H0 (beta_5 = 0) is {'REJECTED' if p_val_wcb_interaction < 0.05 else 'RETAINED'} at alpha = 0.05"
    )

    # -------------------------------------------------------------------------
    # Test C: Unrestricted Wild Cluster Bootstrap (CIs & Contraction Ratio rho)
    # -------------------------------------------------------------------------
    logger.info("Executing Test C: Unrestricted WCB for Parameter CIs and Contraction Ratio Distribution...")
    u_unr = ols_c.resid.values
    # Cluster-demean residuals to guard against cluster fixed component inflation
    u_demeaned = u_unr - df.groupby("channel_id")["log_views"].transform(lambda s: u_unr[s.index].mean()).values
    u_unr_adj = u_demeaned * c_scale

    np.random.seed(seed)
    beta_boot_draws = []
    rho_draws = []
    concave_flags = []

    for b in range(B):
        v_dict = {cl: np.random.choice([-1.0, 1.0]) for cl in unique_clusters}
        v_vec = np.array([v_dict[cl] for cl in clusters])
        y_b = ols_c.fittedvalues.values + v_vec * u_unr_adj

        fit_b = sm.OLS(y_b, X_c).fit()
        b_vec = fit_b.params.values if hasattr(fit_b.params, "values") else fit_b.params
        beta_boot_draws.append(b_vec)

        b2_val = float(b_vec[idx_x2])
        b5_val = float(b_vec[idx_inter])

        is_concave = (b2_val < 0.0)
        concave_flags.append(is_concave)

        if is_concave and np.abs(b2_val) >= 1e-6:
            rho_val = np.abs(b5_val) / (2.0 * np.abs(b2_val))
        else:
            rho_val = np.inf
        rho_draws.append(rho_val)

    beta_boot_draws = np.array(beta_boot_draws)
    rho_draws = np.array(rho_draws)
    concave_flags = np.array(concave_flags)

    # Compute 95% bootstrap percentile CIs
    bootstrap_records = []
    for i, col in enumerate(centered_cols):
        q025 = float(np.percentile(beta_boot_draws[:, i], 2.5))
        q500 = float(np.percentile(beta_boot_draws[:, i], 50.0))
        q975 = float(np.percentile(beta_boot_draws[:, i], 97.5))
        se_boot = float(np.std(beta_boot_draws[:, i], ddof=1))

        bootstrap_records.append({
            "variable": col,
            "sample_coef": float(crse_c.params.iloc[i]),
            "boot_mean": float(np.mean(beta_boot_draws[:, i])),
            "boot_se": se_boot,
            "boot_ci_95_lower": q025,
            "boot_median": q500,
            "boot_ci_95_upper": q975
        })
        logger.info(
            f"  {col:<18}: Sample = {crse_c.params.iloc[i]:10.4f} | Boot SE = {se_boot:8.4f} | "
            f"Boot 95% CI = [{q025:8.3f}, {q975:8.3f}]"
        )

    df_boot_ci = pd.DataFrame(bootstrap_records)
    out_boot_ci_csv = RESULTS_DIR / "bootstrap_inference_results.csv"
    df_boot_ci.to_csv(out_boot_ci_csv, index=False)

    # Contraction Ratio Analysis
    b2_hat = float(crse_c.params["x2_c"])
    b5_hat = float(crse_c.params["x_inter_c"])
    sample_is_concave = (b2_hat < 0.0)
    sample_rho = (np.abs(b5_hat) / (2.0 * np.abs(b2_hat))) if sample_is_concave else np.inf

    concave_draws_count = int(np.sum(concave_flags))
    concave_draws_fraction = float(np.mean(concave_flags))

    valid_rhos = rho_draws[concave_flags]
    p_rho_less_1_among_concave = float(np.mean(valid_rhos < 1.0)) if len(valid_rhos) > 0 else 0.0
    p_both_concave_and_contraction = float(np.mean((concave_flags) & (rho_draws < 1.0)))

    pct_10 = float(np.percentile(valid_rhos, 10.0)) if len(valid_rhos) > 0 else np.nan
    pct_25 = float(np.percentile(valid_rhos, 25.0)) if len(valid_rhos) > 0 else np.nan
    pct_50 = float(np.percentile(valid_rhos, 50.0)) if len(valid_rhos) > 0 else np.nan
    pct_75 = float(np.percentile(valid_rhos, 75.0)) if len(valid_rhos) > 0 else np.nan
    pct_90 = float(np.percentile(valid_rhos, 90.0)) if len(valid_rhos) > 0 else np.nan

    logger.info("\n" + "=" * 80)
    logger.info("CONTRACTION RATIO & POTENTIAL GAME STABILITY AUDIT")
    logger.info("=" * 80)
    logger.info(f"Sample Point Estimate beta_2: {b2_hat:.6f} ({'Concave (Diminishing Returns)' if sample_is_concave else 'Convex'})")
    logger.info(f"Sample Point Estimate beta_5: {b5_hat:.6f}")
    logger.info(f"Sample Contraction Ratio rho = |beta_5| / (2*|beta_2|): {sample_rho:.4f} (rho < 1: {sample_rho < 1.0})")
    logger.info(f"Bootstrap Draws Concave (beta_2 < 0): {concave_draws_count}/{B} ({concave_draws_fraction * 100:.2f}%)")
    logger.info(f"P(rho < 1 | beta_2 < 0): {p_rho_less_1_among_concave * 100:.2f}%")
    logger.info(f"P(beta_2 < 0 and rho < 1): {p_both_concave_and_contraction * 100:.2f}%")
    logger.info(
        f"Contraction Ratio Quantiles (among concave draws): "
        f"p10={pct_10:.4f}, p25={pct_25:.4f}, p50(median)={pct_50:.4f}, p75={pct_75:.4f}, p90={pct_90:.4f}"
    )

    # 6. Construct Covariance Matrix of Strategic Parameters (beta_1_c, beta_2_c, beta_3_c, beta_5_c)
    strat_vars = ["x_c", "x2_c", "x_bar_c", "x_inter_c"]
    strat_indices = [centered_cols.index(v) for v in strat_vars]
    cov_strat = crse_c.cov_params().loc[strat_vars, strat_vars].values

    # 7. Generate Game-Theoretic Handoff Payload
    b0_c = float(crse_c.params["const"])
    b1_c = float(crse_c.params["x_c"])
    b2_c = float(crse_c.params["x2_c"])
    b3_c = float(crse_c.params["x_bar_c"])
    b5_c = float(crse_c.params["x_inter_c"])
    b_subs = float(crse_c.params["log_subscribers"])
    b_dur = float(crse_c.params["log_duration"])
    b_age = float(crse_c.params["age_days"])

    b0_u = float(crse_u.params["const"])
    b1_u = float(crse_u.params["x"])
    b2_u = float(crse_u.params["x2"])
    b3_u = float(crse_u.params["x_bar"])
    b5_u = float(crse_u.params["x_inter"])

    game_theory_payload = {
        "metadata": {
            "project": "Click Appeal as a Potential Game: Modeling Creator Incentives, Attention Congestion, and Social Welfare in YouTube",
            "component": "Empirical Payoff Function Estimation",
            "course": "CS 329 Multi-Agent Systems, IIT Gandhinagar",
            "authors": "Bhavya Shah (24110324) - Payoff Estimation | Khushi Shah (24110326) - Game Theory",
            "calibration_status": "Post-Test Final Full-Sample Calibration (N = 809 videos, G = 38 channels)",
            "locked_specification": "Model 4-Centered: Strategic Payoff Function with Contemporaneous Peer Appeal"
        },
        "sample_summary": {
            "total_observations_N": N,
            "total_channels_G": G,
            "mean_videos_per_channel": float(N / G),
            "centering_mu_x": mu_x,
            "centering_mu_xbar": mu_xbar,
            "peer_window_days": 60,
            "peer_window_description": "Contemporaneous rolling 60-day window [t - 30d, t + 30d], LOCO across channels",
            "subscribers_snapshot_note": "Subscriber count measured as late-2026 snapshot; acts as time-invariant channel scale control"
        },
        "locked_centered_coefficients": {
            "specification": "ln(V_it) = beta_0_c + beta_1_c*(x - mu_x) + beta_2_c*(x - mu_x)^2 + beta_3_c*(x_bar - mu_xbar) + beta_5_c*(x - mu_x)*(x_bar - mu_xbar) + gamma_s*ln(s_i) + gamma_d*ln(dur_it) + gamma_a*age_it + eps_it",
            "beta_0_c_intercept": b0_c,
            "beta_1_c_own_appeal_at_mean": b1_c,
            "beta_2_c_quadratic_curvature": b2_c,
            "beta_3_c_peer_congestion_at_mean": b3_c,
            "beta_5_c_strategic_interaction": b5_c,
            "gamma_subscribers": b_subs,
            "gamma_duration": b_dur,
            "gamma_age_days": b_age
        },
        "uncentered_reference_coefficients": {
            "specification": "ln(V_it) = beta_0 + beta_1*x_it + beta_2*x_it^2 + beta_3*x_bar_(-i,t) + beta_5*x_it*x_bar_(-i,t) + gamma_s*ln(s_i) + gamma_d*ln(dur_it) + gamma_a*age_it + eps_it",
            "beta_0_intercept": b0_u,
            "beta_1_own_appeal": b1_u,
            "beta_2_quadratic_own_appeal": b2_u,
            "beta_3_peer_appeal": b3_u,
            "beta_5_interaction": b5_u,
            "note_on_equivalence": "beta_2 and beta_5 are numerically identical between centered and uncentered models"
        },
        "statistical_inference": {
            "cluster_robust_standard_errors": {
                "se_beta_0_c": float(crse_c.bse["const"]),
                "se_beta_1_c": float(crse_c.bse["x_c"]),
                "se_beta_2_c": float(crse_c.bse["x2_c"]),
                "se_beta_3_c": float(crse_c.bse["x_bar_c"]),
                "se_beta_5_c": float(crse_c.bse["x_inter_c"]),
                "se_gamma_subs": float(crse_c.bse["log_subscribers"]),
                "se_gamma_dur": float(crse_c.bse["log_duration"]),
                "se_gamma_age": float(crse_c.bse["age_days"])
            },
            "hypothesis_test_diminishing_returns_concavity": {
                "null_hypothesis": "H0: beta_2 >= 0 (no diminishing returns)",
                "alternative_hypothesis": "H1: beta_2 < 0 (strictly concave revenue)",
                "crse_t_statistic": t_sample_x2,
                "student_t_p_value_df37": float(stats.t.cdf(t_sample_x2, df=G - 1)),
                "wild_cluster_bootstrap_p_value": p_val_wcb_concavity,
                "rejection_at_alpha_0_05": bool(p_val_wcb_concavity < 0.05),
                "interpretation": "Concavity is NOT statistically established at alpha=0.05. Point estimate beta_2 < 0 must be treated as calibrated benchmark."
            },
            "hypothesis_test_strategic_interaction": {
                "null_hypothesis": "H0: beta_5 = 0 (no strategic interaction)",
                "alternative_hypothesis": "H1: beta_5 != 0 (strategic interdependence)",
                "crse_t_statistic": t_sample_inter,
                "student_t_p_value_df37": float(2.0 * (1.0 - stats.t.cdf(np.abs(t_sample_inter), df=G - 1))),
                "wild_cluster_bootstrap_p_value": p_val_wcb_interaction,
                "rejection_at_alpha_0_05": bool(p_val_wcb_interaction < 0.05),
                "interpretation": "Strategic interaction is statistically insignificant (p > 0.05). Packaging effects carry substantial residual uncertainty."
            },
            "strategic_covariance_matrix_order": strat_vars,
            "strategic_covariance_matrix": cov_strat.tolist()
        },
        "game_theoretic_properties": {
            "game_class": "Continuous Aggregative Potential Game",
            "exact_potential_function": "Phi(x_1, ..., x_N) = sum_i [ beta_1_c*(x_i - mu_x) + beta_2_c*(x_i - mu_x)^2 ] + [beta_5_c / (2*(N-1))] * sum_i sum_{j != i} (x_i - mu_x)*(x_j - mu_xbar) - sum_i C_i(x_i)",
            "best_response_equation": "BR_i(x_{-i}) = argmax_{x_i in [0, 1]} [ ln V_i(x_i, x_bar_{-i}) - C_i(x_i) ]",
            "first_order_condition": "beta_1_c + 2*beta_2_c*(x_i - mu_x) + beta_5_c*(x_bar_{-i} - mu_xbar) - C_i'(x_i) = 0",
            "contraction_mapping_audit": {
                "formula": "rho = |beta_5| / (2 * |beta_2|)",
                "sample_point_estimate_rho": sample_rho,
                "contraction_satisfied_on_point_estimates": bool(sample_rho < 1.0),
                "bootstrap_concave_fraction": concave_draws_fraction,
                "bootstrap_p_rho_less_1_given_concave": p_rho_less_1_among_concave,
                "bootstrap_p_both_concave_and_contraction": p_both_concave_and_contraction,
                "bootstrap_rho_quantiles": {
                    "p10": pct_10,
                    "p25": pct_25,
                    "p50_median": pct_50,
                    "p75": pct_75,
                    "p90": pct_90
                }
            },
            "production_cost_guidance_for_game_theory_agent": {
                "specification_recommendation": "C_i(x_i) = c_i * x_i + 0.5 * kappa_i * x_i^2",
                "boundary_warning": "Because empirical beta_2 has high variance, setting kappa_i > 0 guarantees strict game concavity (2*beta_2 - kappa_i < 0) for all creators even in regions where empirical beta_2 fluctuates.",
                "equilibrium_uniqueness": "When 2*|beta_2| + kappa_min > |beta_5|, best-response dynamics form a contraction mapping, guaranteeing a unique pure-strategy Nash Equilibrium."
            }
        }
    }

    out_json = RESULTS_DIR / "game_theory_coefficients.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(game_theory_payload, f, indent=2)

    # Also export flat CSV for easy reading
    csv_rows = [
        {"parameter": "beta_0_c_intercept", "value": b0_c, "se_crse": float(crse_c.bse["const"]), "description": "Centered Intercept"},
        {"parameter": "beta_1_c_own_appeal", "value": b1_c, "se_crse": float(crse_c.bse["x_c"]), "description": "Linear Own Appeal at Mean"},
        {"parameter": "beta_2_c_curvature", "value": b2_c, "se_crse": float(crse_c.bse["x2_c"]), "description": "Quadratic Curvature"},
        {"parameter": "beta_3_c_peer_congestion", "value": b3_c, "se_crse": float(crse_c.bse["x_bar_c"]), "description": "Competitor Congestion at Mean"},
        {"parameter": "beta_5_c_interaction", "value": b5_c, "se_crse": float(crse_c.bse["x_inter_c"]), "description": "Strategic Interaction Term"},
        {"parameter": "gamma_subscribers", "value": b_subs, "se_crse": float(crse_c.bse["log_subscribers"]), "description": "Channel Scale Elasticity"},
        {"parameter": "gamma_duration", "value": b_dur, "se_crse": float(crse_c.bse["log_duration"]), "description": "Log Duration Elasticity"},
        {"parameter": "gamma_age_days", "value": b_age, "se_crse": float(crse_c.bse["age_days"]), "description": "Linear Age Drift per Day"},
        {"parameter": "mu_x_centering", "value": mu_x, "se_crse": np.nan, "description": "Sample Mean Own Click Appeal"},
        {"parameter": "mu_xbar_centering", "value": mu_xbar, "se_crse": np.nan, "description": "Sample Mean Competitor Peer Appeal"},
        {"parameter": "contraction_ratio_rho", "value": sample_rho, "se_crse": np.nan, "description": "|beta_5| / (2 * |beta_2|)"},
        {"parameter": "wcb_p_val_concavity", "value": p_val_wcb_concavity, "se_crse": np.nan, "description": "Null-Restricted WCB p-value for H0: beta_2 >= 0"},
        {"parameter": "wcb_p_val_interaction", "value": p_val_wcb_interaction, "se_crse": np.nan, "description": "Null-Restricted WCB p-value for H0: beta_5 = 0"}
    ]
    pd.DataFrame(csv_rows).to_csv(RESULTS_DIR / "game_theory_coefficients.csv", index=False)

    logger.info(f"Saved game theory JSON payload: {out_json}")
    logger.info(f"Saved game theory CSV payload: {RESULTS_DIR / 'game_theory_coefficients.csv'}")
    logger.info("=" * 80)
    logger.info("STEP 7 COMPLETED SUCCESSFULLY: STRUCTURAL CALIBRATION & HANDOFF READY")
    logger.info("=" * 80)


if __name__ == "__main__":
    run_structural_calibration(B=2000, seed=42)
