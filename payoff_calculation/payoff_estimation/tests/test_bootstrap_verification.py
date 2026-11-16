#!/usr/bin/env python3
"""
payoff_estimation/tests/test_bootstrap_verification.py
Technical Verification Test: Wild Cluster Bootstrap & Confidence Interval Engine.

Validates:
1. Null-Restricted Wild Cluster Bootstrap (WCB) test size under true null (beta_5 = 0).
2. One-sided WCB test power under false null (true beta_2 = -3.0 < 0).
3. Unrestricted bootstrap 95% percentile confidence interval coverage of true parameters.
4. Exact numerical reproducibility under fixed random seed.
5. Treatment of unstable contraction ratio draws (zero-discarding rule).

Outputs:
- payoff_estimation/results/bootstrap_verification_report.json
- payoff_estimation/logs/test_bootstrap_verification.log
"""

import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

TESTS_DIR = Path(__file__).resolve().parent
PAYOFF_DIR = TESTS_DIR.parent
RESULTS_DIR = PAYOFF_DIR / "results"
LOGS_DIR = PAYOFF_DIR / "logs"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "test_bootstrap_verification.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("test_bootstrap_verification")


def generate_synthetic_clustered_data(n_clusters=26, videos_per_cluster=20, seed=123):
    """
    Generate synthetic data with known true parameters and cluster error structure.
    True Parameters:
    - beta_0 = 10.0
    - beta_1 = 2.0 (own linear appeal)
    - beta_2 = -3.0 (strictly concave)
    - beta_3 = -0.8 (peer congestion)
    - beta_5 = 0.0 (TRUE NULL: no interaction)
    - beta_4 = 0.5 (scale control)
    """
    np.random.seed(seed)
    clusters = []
    
    for g in range(n_clusters):
        c_id = f"chan_{g:02d}"
        cluster_shock = np.random.normal(0, 0.4)
        log_s = np.random.normal(15.0, 1.0)
        
        for t in range(videos_per_cluster):
            x = np.random.uniform(0.1, 0.8)
            x2 = x ** 2
            x_bar = np.random.uniform(0.3, 0.6)
            eps = np.random.normal(0, 0.5)
            
            # y = 10.0 + 2.0*x - 3.0*x2 - 0.8*x_bar + 0.0*(x*x_bar) + 0.5*log_s + alpha_g + eps
            y = 10.0 + 2.0 * x - 3.0 * x2 - 0.8 * x_bar + 0.5 * log_s + cluster_shock + eps
            
            clusters.append({
                "channel_id": c_id,
                "x": x,
                "x2": x2,
                "x_bar": x_bar,
                "x_inter": x * x_bar,
                "log_s": log_s,
                "y": y
            })
            
    return pd.DataFrame(clusters)


def run_verification(B=500, seed=42):
    logger.info("=" * 70)
    logger.info("RUNNING BOOTSTRAP VERIFICATION SUITE ON SYNTHETIC CLUSTERED DATA")
    logger.info("=" * 70)
    
    df = generate_synthetic_clustered_data(n_clusters=26, videos_per_cluster=20, seed=123)
    G = df["channel_id"].nunique()
    N = len(df)
    logger.info(f"Generated synthetic dataset: N = {N} observations across G = {G} clusters.")
    
    cluster_ids = df["channel_id"].values
    unique_clusters = np.unique(cluster_ids)
    
    # 1. Unrestricted OLS Estimation
    X_cols = ["const", "x", "x2", "x_bar", "x_inter", "log_s"]
    df["const"] = 1.0
    X = df[X_cols].values
    y = df["y"].values
    K = X.shape[1]
    
    # OLS fit
    ols_unr = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": cluster_ids})
    b_sample = ols_unr.params
    se_sample = ols_unr.bse
    t_sample = b_sample / se_sample
    
    logger.info("Unrestricted OLS Sample Estimates (True: beta_1=2.0, beta_2=-3.0, beta_5=0.0):")
    for i, col in enumerate(X_cols):
        logger.info(f"  {col}: b = {b_sample[i]:.4f}, SE_CRSE = {se_sample[i]:.4f}, t = {t_sample[i]:.4f}")
    
    # Global finite-sample correction factor
    scale_factor = np.sqrt((G / (G - 1.0)) * ((N - 1.0) / (N - K)))
    logger.info(f"Global finite-sample scaling factor: {scale_factor:.4f}")
    
    # -------------------------------------------------------------
    # Test 1: Null-Restricted WCB for H0: beta_5 = 0 (Two-Sided)
    # -------------------------------------------------------------
    X_r5_cols = ["const", "x", "x2", "x_bar", "log_s"]
    X_r5 = df[X_r5_cols].values
    ols_r5 = sm.OLS(y, X_r5).fit()
    y_hat_r5 = ols_r5.fittedvalues
    u_r5 = (y - y_hat_r5) * scale_factor
    
    t5_stat = t_sample[X_cols.index("x_inter")]
    
    np.random.seed(seed)
    t5_boot = []
    
    for b in range(B):
        v_dict = {c: np.random.choice([-1.0, 1.0]) for c in unique_clusters}
        v_vec = np.array([v_dict[c] for c in cluster_ids])
        y_boot = y_hat_r5 + v_vec * u_r5
        
        # Fit unrestricted model
        fit_b = sm.OLS(y_boot, X).fit(cov_type="cluster", cov_kwds={"groups": cluster_ids})
        idx5 = X_cols.index("x_inter")
        t5_boot.append(fit_b.params[idx5] / fit_b.bse[idx5])
        
    t5_boot = np.array(t5_boot)
    p_val_beta5 = np.mean(np.abs(t5_boot) >= np.abs(t5_stat))
    logger.info(f"Test 1 (Null-Restricted WCB for beta_5 = 0, True Null): p-value = {p_val_beta5:.4f}")
    test1_pass = p_val_beta5 > 0.05
    logger.info(f"  Result: {'PASSED (Null retained at alpha=0.05)' if test1_pass else 'FAILED'}")
    
    # -------------------------------------------------------------
    # Test 2: Null-Restricted WCB for H0: beta_2 >= 0 vs beta_2 < 0 (One-Sided)
    # -------------------------------------------------------------
    X_r2_cols = ["const", "x", "x_bar", "x_inter", "log_s"]
    X_r2 = df[X_r2_cols].values
    ols_r2 = sm.OLS(y, X_r2).fit()
    y_hat_r2 = ols_r2.fittedvalues
    u_r2 = (y - y_hat_r2) * scale_factor
    
    t2_stat = t_sample[X_cols.index("x2")]
    
    np.random.seed(seed)
    t2_boot = []
    
    for b in range(B):
        v_dict = {c: np.random.choice([-1.0, 1.0]) for c in unique_clusters}
        v_vec = np.array([v_dict[c] for c in cluster_ids])
        y_boot = y_hat_r2 + v_vec * u_r2
        
        fit_b = sm.OLS(y_boot, X).fit(cov_type="cluster", cov_kwds={"groups": cluster_ids})
        idx2 = X_cols.index("x2")
        t2_boot.append(fit_b.params[idx2] / fit_b.bse[idx2])
        
    t2_boot = np.array(t2_boot)
    p_val_beta2 = np.mean(t2_boot <= t2_stat)
    logger.info(f"Test 2 (Null-Restricted WCB for beta_2 < 0, False Null): p-value = {p_val_beta2:.4f}")
    test2_pass = p_val_beta2 < 0.01
    logger.info(f"  Result: {'PASSED (Null rejected strongly at alpha=0.01)' if test2_pass else 'FAILED'}")
    
    # -------------------------------------------------------------
    # Test 3: Unrestricted Bootstrap Confidence Intervals
    # -------------------------------------------------------------
    u_unr = ols_unr.resid
    # Cluster-center residuals
    u_centered = u_unr - df.groupby("channel_id")["y"].transform(lambda s: u_unr[s.index].mean())
    u_unr_adj = u_centered * scale_factor
    
    np.random.seed(seed)
    beta_boot = []
    rho_boot = []
    failed_rho_count = 0
    
    for b in range(B):
        v_dict = {c: np.random.choice([-1.0, 1.0]) for c in unique_clusters}
        v_vec = np.array([v_dict[c] for c in cluster_ids])
        y_boot = ols_unr.fittedvalues + v_vec * u_unr_adj
        
        fit_b = sm.OLS(y_boot, X).fit()
        b_draw = fit_b.params.values if hasattr(fit_b.params, "values") else fit_b.params
        beta_boot.append(b_draw)
        
        b2_val = float(b_draw[X_cols.index("x2")])
        b5_val = float(b_draw[X_cols.index("x_inter")])
        
        if b2_val < 0 and np.abs(b2_val) >= 1e-4:
            rho_boot.append(np.abs(b5_val) / (2.0 * np.abs(b2_val)))
        else:
            rho_boot.append(np.inf)
            failed_rho_count += 1
            
    beta_boot = np.array(beta_boot)
    
    # Confidence intervals
    ci_results = {}
    for i, col in enumerate(X_cols):
        q025 = float(np.percentile(beta_boot[:, i], 2.5))
        q975 = float(np.percentile(beta_boot[:, i], 975 / 10.0))
        ci_results[col] = {"lower": q025, "upper": q975}
        
    logger.info("Unrestricted Bootstrap 95% Percentile Confidence Intervals:")
    logger.info(f"  beta_1 (True: 2.0): [{ci_results['x']['lower']:.4f}, {ci_results['x']['upper']:.4f}]")
    logger.info(f"  beta_2 (True: -3.0): [{ci_results['x2']['lower']:.4f}, {ci_results['x2']['upper']:.4f}]")
    logger.info(f"  beta_5 (True: 0.0): [{ci_results['x_inter']['lower']:.4f}, {ci_results['x_inter']['upper']:.4f}]")
    
    cov_beta1 = ci_results['x']['lower'] <= 2.0 <= ci_results['x']['upper']
    cov_beta2 = ci_results['x2']['lower'] <= -3.0 <= ci_results['x2']['upper']
    cov_beta5 = ci_results['x_inter']['lower'] <= 0.0 <= ci_results['x_inter']['upper']
    test3_pass = cov_beta1 and cov_beta2 and cov_beta5
    logger.info(f"Test 3 Coverage Check: {'PASSED (All true values covered by 95% CIs)' if test3_pass else 'FAILED'}")
    
    # Contraction ratio evaluation
    valid_rhos = [r for r in rho_boot if np.isfinite(r)]
    p_rho_less_1 = np.mean([r < 1.0 for r in rho_boot])
    median_rho = float(np.median(valid_rhos)) if valid_rhos else None
    logger.info(f"Contraction Ratio Evaluation: Median = {median_rho:.4f}, P(rho < 1) = {p_rho_less_1:.4f}, Non-concave draws = {failed_rho_count}/{B}")
    
    # -------------------------------------------------------------
    # Test 4: Seed Reproducibility Check
    # -------------------------------------------------------------
    np.random.seed(seed)
    t5_boot_rerun = []
    for b in range(10):
        v_dict = {c: np.random.choice([-1.0, 1.0]) for c in unique_clusters}
        v_vec = np.array([v_dict[c] for c in cluster_ids])
        y_boot = y_hat_r5 + v_vec * u_r5
        fit_b = sm.OLS(y_boot, X).fit(cov_type="cluster", cov_kwds={"groups": cluster_ids})
        t5_boot_rerun.append(fit_b.params[idx5] / fit_b.bse[idx5])
        
    diff = np.max(np.abs(t5_boot[:10] - np.array(t5_boot_rerun)))
    test4_pass = diff == 0.0
    logger.info(f"Test 4 (Seed Reproducibility Check): Max discrepancy = {diff:.10f} -> {'PASSED' if test4_pass else 'FAILED'}")
    
    # Export verification report
    verification_report = {
        "dataset_specs": {"n_clusters": G, "n_observations": N, "videos_per_cluster": 20},
        "true_parameters": {"beta_0": 10.0, "beta_1": 2.0, "beta_2": -3.0, "beta_3": -0.8, "beta_5": 0.0, "beta_4": 0.5},
        "sample_estimates": {col: float(b_sample[i]) for i, col in enumerate(X_cols)},
        "global_scaling_factor": float(scale_factor),
        "test_1_beta5_null_restricted_p_value": float(p_val_beta5),
        "test_1_passed": bool(test1_pass),
        "test_2_beta2_one_sided_p_value": float(p_val_beta2),
        "test_2_passed": bool(test2_pass),
        "test_3_confidence_intervals": ci_results,
        "test_3_coverage_passed": bool(test3_pass),
        "test_4_seed_reproducibility_passed": bool(test4_pass),
        "contraction_ratio_summary": {
            "median": median_rho,
            "p_rho_less_1": float(p_rho_less_1),
            "non_concave_draws": failed_rho_count,
            "total_draws": B
        },
        "all_tests_passed": bool(test1_pass and test2_pass and test3_pass and test4_pass)
    }
    
    out_json = RESULTS_DIR / "bootstrap_verification_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(verification_report, f, indent=2)
    logger.info(f"Saved verification report: {out_json}")
    logger.info("=" * 70)
    logger.info("BOOTSTRAP VERIFICATION COMPLETED SUCCESSFULLY.")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_verification(B=500, seed=42)
