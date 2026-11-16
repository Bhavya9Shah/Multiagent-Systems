#!/usr/bin/env python3
"""
payoff_estimation/scripts/04_evaluate_loco_cv.py
Phase 2 - Step 4: Leak-Free Leave-One-Channel-Out Cross-Validation (LOCO-CV).

Executed strictly within the 26 Training Channels (N = 541 observations, 26 folds).

Key Methodological Guarantees:
1. Complete isolation: Validation (6 channels) and Test (6 channels) are NEVER accessed.
2. Leak-free peer construction:
   - For each fold k (channel k held out):
     * Training peer averages are computed strictly among the other 24 active training channels.
     * Held-out channel k's peer averages are computed strictly using the 25 active training channels.
3. Fold-specific preprocessing:
   - Centering means mu_x^(k) and mu_xbar^(k) are computed strictly from the 25 active training channels.
   - Held-out channel k's features are centered using the training fold's means.
4. Candidate Models Evaluated:
   - M0: Baseline Controls (log(S), log(duration), age_days)
   - M1: Linear Appeal (x_c) + Controls
   - M2: Quadratic Appeal (x_c, x2_c) + Controls
   - M3: Peer Congestion (x_c, x2_c, x_bar_c) + Controls
   - M4: Strategic Interaction (x_c, x2_c, x_bar_c, x_inter_c) + Controls
   - M5: Full Specification with Tags (M4 + log(tags + 1))
5. Metrics:
   - Fold-level: Channel MAE, RMSE, Mean Bias, SSE, SST
   - Pooled Aggregate: Pooled R²_OOS = 1 - (SSE_OOS / SST_OOS), Pooled RMSE_OOS, Pooled MAE_OOS

Outputs:
- payoff_estimation/results/loco_cv_fold_results.csv
- payoff_estimation/results/loco_cv_aggregate_summary.csv
- payoff_estimation/logs/04_evaluate_loco_cv.log
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PAYOFF_DIR = SCRIPT_DIR.parent
DATA_DIR = PAYOFF_DIR / "data"
RESULTS_DIR = PAYOFF_DIR / "results"
LOGS_DIR = PAYOFF_DIR / "logs"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
LOG_FILE = LOGS_DIR / "04_evaluate_loco_cv.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("04_evaluate_loco_cv")


def run_loco_cv():
    logger.info("=" * 70)
    logger.info("STEP 4: LEAK-FREE 26-FOLD WITHIN-TRAIN LOCO-CV")
    logger.info("=" * 70)

    input_csv = DATA_DIR / "partitioned_payoff_data.csv"
    if not input_csv.exists():
        logger.error(f"Cannot find partitioned dataset: {input_csv}")
        sys.exit(1)

    full_df = pd.read_csv(input_csv)
    train_df = full_df[full_df["partition"] == "train"].copy()
    train_df["published_at_dt"] = pd.to_datetime(train_df["published_at"])

    N_train = len(train_df)
    unique_channels = train_df["channel_id"].unique()
    G_train = len(unique_channels)

    logger.info(f"Loaded Training Sample: N = {N_train} videos across G = {G_train} channels.")
    assert N_train == 541, f"Expected 541 training videos, got {N_train}"
    assert G_train == 26, f"Expected 26 training channels, got {G_train}"

    # Model specifications
    models = {
        "Model_0_Baseline": ["const", "log_subscribers", "log_duration", "age_days"],
        "Model_1_Linear": ["const", "x_c", "log_subscribers", "log_duration", "age_days"],
        "Model_2_Quadratic": ["const", "x_c", "x2_c", "log_subscribers", "log_duration", "age_days"],
        "Model_3_PeerCongestion": ["const", "x_c", "x2_c", "x_bar_c", "log_subscribers", "log_duration", "age_days"],
        "Model_4_Interaction": ["const", "x_c", "x2_c", "x_bar_c", "x_inter_c", "log_subscribers", "log_duration", "age_days"],
        "Model_5_FullTags": ["const", "x_c", "x2_c", "x_bar_c", "x_inter_c", "log_subscribers", "log_duration", "age_days", "log_tags_plus1"]
    }

    fold_records = []
    pooled_actuals = []
    pooled_benchmark_means = []
    pooled_preds = {m: [] for m in models}

    logger.info("Beginning 26 cross-validation iterations (Leave-One-Channel-Out)...")

    for fold_idx, held_out_id in enumerate(unique_channels, 1):
        train_fold = train_df[train_df["channel_id"] != held_out_id].copy()
        test_fold = train_df[train_df["channel_id"] == held_out_id].copy()

        ch_name = test_fold["channel_name"].iloc[0]
        tier = test_fold["subscriber_tier"].iloc[0]
        n_test_vids = len(test_fold)

        # 1. Fold-specific training peer construction
        # Competitors strictly from the other 24 training channels
        fold_train_peers = []
        for _, row in train_fold.iterrows():
            c_id = row["channel_id"]
            t_pub = row["published_at_dt"]
            w = train_fold[
                (train_fold["channel_id"] != c_id) &
                (train_fold["published_at_dt"] >= t_pub - pd.Timedelta(days=30)) &
                (train_fold["published_at_dt"] <= t_pub + pd.Timedelta(days=30))
            ]
            if len(w) > 0:
                fold_train_peers.append(float(w.groupby("channel_id")["x"].mean().mean()))
            else:
                loco = train_fold[train_fold["channel_id"] != c_id]
                fold_train_peers.append(float(loco.groupby("channel_id")["x"].mean().mean()))

        train_fold["x_bar"] = fold_train_peers

        # 2. Fold-specific centering parameters
        mu_x_k = float(train_fold["x"].mean())
        mu_xbar_k = float(train_fold["x_bar"].mean())
        y_train_mean_k = float(train_fold["log_views"].mean())

        train_fold["x_c"] = train_fold["x"] - mu_x_k
        train_fold["x2_c"] = train_fold["x_c"] ** 2
        train_fold["x_bar_c"] = train_fold["x_bar"] - mu_xbar_k
        train_fold["x_inter_c"] = train_fold["x_c"] * train_fold["x_bar_c"]
        train_fold["const"] = 1.0

        # 3. Peer construction for held-out channel k from the 25 training channels
        fold_test_peers = []
        for _, row in test_fold.iterrows():
            t_pub = row["published_at_dt"]
            w = train_fold[
                (train_fold["published_at_dt"] >= t_pub - pd.Timedelta(days=30)) &
                (train_fold["published_at_dt"] <= t_pub + pd.Timedelta(days=30))
            ]
            if len(w) > 0:
                fold_test_peers.append(float(w.groupby("channel_id")["x"].mean().mean()))
            else:
                fold_test_peers.append(float(train_fold.groupby("channel_id")["x"].mean().mean()))

        test_fold["x_bar"] = fold_test_peers

        # Center held-out features with fold training means
        test_fold["x_c"] = test_fold["x"] - mu_x_k
        test_fold["x2_c"] = test_fold["x_c"] ** 2
        test_fold["x_bar_c"] = test_fold["x_bar"] - mu_xbar_k
        test_fold["x_inter_c"] = test_fold["x_c"] * test_fold["x_bar_c"]
        test_fold["const"] = 1.0

        y_actual_k = test_fold["log_views"].values
        pooled_actuals.extend(y_actual_k)
        pooled_benchmark_means.extend([y_train_mean_k] * n_test_vids)

        fold_row = {
            "fold": fold_idx,
            "channel_id": held_out_id,
            "channel_name": ch_name,
            "subscriber_tier": tier,
            "n_videos": n_test_vids,
            "y_train_mean": y_train_mean_k,
            "mu_x_train": mu_x_k,
            "mu_xbar_train": mu_xbar_k
        }

        # Fit each model and predict
        for m_name, cols in models.items():
            fit = sm.OLS(train_fold["log_views"], train_fold[cols]).fit()
            y_pred_k = fit.predict(test_fold[cols]).values
            pooled_preds[m_name].extend(y_pred_k)

            # Fold-level performance metrics
            err_k = y_actual_k - y_pred_k
            mae_k = float(np.mean(np.abs(err_k)))
            rmse_k = float(np.sqrt(np.mean(err_k ** 2)))
            bias_k = float(np.mean(err_k))
            sse_k = float(np.sum(err_k ** 2))

            fold_row[f"{m_name}_mae"] = mae_k
            fold_row[f"{m_name}_rmse"] = rmse_k
            fold_row[f"{m_name}_bias"] = bias_k
            fold_row[f"{m_name}_sse"] = sse_k

        fold_records.append(fold_row)

        if fold_idx % 5 == 0 or fold_idx == G_train:
            logger.info(f"Completed Fold {fold_idx:02d}/{G_train:02d}: {ch_name} ({n_test_vids} vids)")

    # 4. Compute Pooled Aggregate Metrics Across All 541 Held-Out Observations
    y_actuals_arr = np.array(pooled_actuals)
    y_benchmarks_arr = np.array(pooled_benchmark_means)
    SST_OOS = float(np.sum((y_actuals_arr - y_benchmarks_arr) ** 2))

    summary_records = []
    logger.info("\n" + "=" * 70)
    logger.info("POOLED LEAVE-ONE-CHANNEL-OUT (LOCO) CROSS-VALIDATION SUMMARY:")
    logger.info(f"Total Evaluated Observations: N = {len(y_actuals_arr)}, SST_OOS = {SST_OOS:.2f}")
    logger.info("=" * 70)

    for m_name in models:
        y_preds_arr = np.array(pooled_preds[m_name])
        SSE_OOS = float(np.sum((y_actuals_arr - y_preds_arr) ** 2))
        R2_OOS = 1.0 - (SSE_OOS / SST_OOS)
        RMSE_OOS = float(np.sqrt(np.mean((y_actuals_arr - y_preds_arr) ** 2)))
        MAE_OOS = float(np.mean(np.abs(y_actuals_arr - y_preds_arr)))

        # Channel-level dispersion
        channel_maes = [r[f"{m_name}_mae"] for r in fold_records]
        mean_ch_mae = float(np.mean(channel_maes))
        median_ch_mae = float(np.median(channel_maes))
        std_ch_mae = float(np.std(channel_maes))
        iqr_ch_mae = float(np.percentile(channel_maes, 75) - np.percentile(channel_maes, 25))

        summary_records.append({
            "model_name": m_name,
            "n_folds": G_train,
            "n_observations": len(y_actuals_arr),
            "sse_oos": SSE_OOS,
            "sst_oos": SST_OOS,
            "r2_oos": R2_OOS,
            "rmse_oos": RMSE_OOS,
            "mae_oos": MAE_OOS,
            "mean_channel_mae": mean_ch_mae,
            "median_channel_mae": median_ch_mae,
            "std_channel_mae": std_ch_mae,
            "iqr_channel_mae": iqr_ch_mae
        })

        logger.info(
            f"{m_name:<25}: R²_OOS = {R2_OOS:.4f} | RMSE = {RMSE_OOS:.4f} | MAE = {MAE_OOS:.4f} | "
            f"Median Ch MAE = {median_ch_mae:.4f} (IQR = {iqr_ch_mae:.4f})"
        )

    # 5. Export Tables
    df_folds = pd.DataFrame(fold_records)
    df_summary = pd.DataFrame(summary_records)

    out_folds_csv = RESULTS_DIR / "loco_cv_fold_results.csv"
    out_summary_csv = RESULTS_DIR / "loco_cv_aggregate_summary.csv"

    df_folds.to_csv(out_folds_csv, index=False)
    df_summary.to_csv(out_summary_csv, index=False)

    logger.info(f"Saved fold-level LOCO-CV results: {out_folds_csv}")
    logger.info(f"Saved aggregate LOCO-CV summary: {out_summary_csv}")
    logger.info("=" * 70)
    logger.info("STEP 4 COMPLETED SUCCESSFULLY: LOCO-CV EVALUATED ON 26 TRAINING FOLDS")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_loco_cv()
