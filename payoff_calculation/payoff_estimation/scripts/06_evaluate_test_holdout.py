#!/usr/bin/env python3
"""
payoff_estimation/scripts/06_evaluate_test_holdout.py
Phase 2 - Step 6: Final Holdout Evaluation on Untouched Test Channels (6 channels, 132 videos).

Requirements:
1. Conduct final evaluation on the 6 held-out test channels (N = 132 videos)
   after all candidate models have been evaluated on validation and the final specification
   (Model 4-Centered) has been locked.
2. Refit the locked specification on the combined training + validation sample (32 channels, 677 videos)
   before generating test predictions.
3. Test peer appeal is constructed from the combined training + validation sample (within [t-30d, t+30d]).
4. Centering means (mu_x, mu_xbar) and baseline mean y are computed from the combined training + validation sample.
5. Compute and report test OOS R², RMSE, MAE, and channel-by-channel error dispersion:
   (Airrack, Foodie We, How Ridiculous, Joel Hansen, Matt Stonie, Stokes Twins).
6. Disclose prior exploratory Phase 1 history: all 809 observations were analyzed in exploratory analyses
   before the train/validation/test split protocol was established, so this evaluation is an internal
   holdout validation of the locked specification rather than a pure pre-registered ex-ante trial.

Outputs:
- payoff_estimation/results/test_set_evaluation.csv
- payoff_estimation/results/test_channel_dispersion.csv
- payoff_estimation/logs/06_evaluate_test_holdout.log
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
LOG_FILE = LOGS_DIR / "06_evaluate_test_holdout.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("06_evaluate_test_holdout")


def run_test_evaluation():
    logger.info("=" * 75)
    logger.info("STEP 6: FINAL HOLDOUT EVALUATION ON 6 TEST CHANNELS (132 VIDEOS)")
    logger.info("=" * 75)
    logger.info(
        "METHODOLOGICAL DISCLOSURE:\n"
        "All 809 observations were analyzed during initial exploratory Phase 1 analyses before\n"
        "the channel-stratified train/validation/test split protocol was formalized. Therefore,\n"
        "this test evaluation serves as an internal holdout generalization check of the locked\n"
        "model specification (Model 4-Centered) rather than a pure ex-ante pre-registered trial."
    )

    input_csv = DATA_DIR / "partitioned_payoff_data.csv"
    if not input_csv.exists():
        logger.error(f"Cannot find partitioned dataset: {input_csv}")
        sys.exit(1)

    full_df = pd.read_csv(input_csv)
    full_df["published_at_dt"] = pd.to_datetime(full_df["published_at"])

    train_val_df = full_df[full_df["partition"].isin(["train", "validation"])].copy()
    test_df = full_df[full_df["partition"] == "test"].copy()

    N_trainval = len(train_val_df)
    G_trainval = train_val_df["channel_id"].nunique()
    N_test = len(test_df)
    G_test = test_df["channel_id"].nunique()

    logger.info(f"Combined Estimation Sample (Train + Val): N = {N_trainval} videos across G = {G_trainval} channels.")
    logger.info(f"Held-Out Test Sample: N = {N_test} videos across G = {G_test} channels.")
    assert N_trainval == 677, f"Expected 677 train+val videos, got {N_trainval}"
    assert G_trainval == 32, f"Expected 32 train+val channels, got {G_trainval}"
    assert N_test == 132, f"Expected 132 test videos, got {N_test}"
    assert G_test == 6, f"Expected 6 test channels, got {G_test}"

    # 1. Peer Appeal for Train+Val (LOCO within Train+Val)
    logger.info("Constructing peer appeal for Train+Val sample (LOCO across 32 channels)...")
    trainval_peers = []
    for _, row in train_val_df.iterrows():
        c_id = row["channel_id"]
        t_pub = row["published_at_dt"]
        w = train_val_df[
            (train_val_df["channel_id"] != c_id) &
            (train_val_df["published_at_dt"] >= t_pub - pd.Timedelta(days=30)) &
            (train_val_df["published_at_dt"] <= t_pub + pd.Timedelta(days=30))
        ]
        if len(w) > 0:
            trainval_peers.append(float(w.groupby("channel_id")["x"].mean().mean()))
        else:
            loco = train_val_df[train_val_df["channel_id"] != c_id]
            trainval_peers.append(float(loco.groupby("channel_id")["x"].mean().mean()))
    train_val_df["x_bar"] = trainval_peers

    # 2. Peer Appeal for Test Sample (strictly from the 32 Train+Val channels)
    logger.info("Constructing peer appeal for Test sample (strictly from 32 Train+Val channels)...")
    test_peers = []
    for _, row in test_df.iterrows():
        t_pub = row["published_at_dt"]
        w = train_val_df[
            (train_val_df["published_at_dt"] >= t_pub - pd.Timedelta(days=30)) &
            (train_val_df["published_at_dt"] <= t_pub + pd.Timedelta(days=30))
        ]
        if len(w) > 0:
            test_peers.append(float(w.groupby("channel_id")["x"].mean().mean()))
        else:
            test_peers.append(float(train_val_df.groupby("channel_id")["x"].mean().mean()))
    test_df["x_bar"] = test_peers

    # 3. Centering parameters (Strictly from 32 Train+Val channels)
    mu_x_trainval = float(train_val_df["x"].mean())
    mu_xbar_trainval = float(train_val_df["x_bar"].mean())
    y_trainval_mean = float(train_val_df["log_views"].mean())

    logger.info(
        f"Train+Val Centering Parameters: mu_x = {mu_x_trainval:.6f}, "
        f"mu_xbar = {mu_xbar_trainval:.6f}, y_trainval_mean = {y_trainval_mean:.6f}"
    )

    for d in [train_val_df, test_df]:
        d["x_c"] = d["x"] - mu_x_trainval
        d["x2_c"] = d["x_c"] ** 2
        d["x_bar_c"] = d["x_bar"] - mu_xbar_trainval
        d["x_inter_c"] = d["x_c"] * d["x_bar_c"]
        d["const"] = 1.0

    # 4. Models to evaluate
    models = {
        "Model_0_Baseline": ["const", "log_subscribers", "log_duration", "age_days"],
        "Model_1_Linear": ["const", "x_c", "log_subscribers", "log_duration", "age_days"],
        "Model_2_Quadratic": ["const", "x_c", "x2_c", "log_subscribers", "log_duration", "age_days"],
        "Model_3_PeerCongestion": ["const", "x_c", "x2_c", "x_bar_c", "log_subscribers", "log_duration", "age_days"],
        "Model_4_Interaction": ["const", "x_c", "x2_c", "x_bar_c", "x_inter_c", "log_subscribers", "log_duration", "age_days"],
        "Model_5_FullTags": ["const", "x_c", "x2_c", "x_bar_c", "x_inter_c", "log_subscribers", "log_duration", "age_days", "log_tags_plus1"]
    }

    y_test = test_df["log_views"].values
    sst_test = float(np.sum((y_test - y_trainval_mean) ** 2))

    test_summary = []
    test_channel_records = []

    logger.info("\nRefitting on Train+Val (32 Channels) and Evaluating on Test (6 Channels)...")

    for m_name, cols in models.items():
        # Fit on 32 Train+Val channels
        fit = sm.OLS(train_val_df["log_views"], train_val_df[cols]).fit(
            cov_type="cluster", cov_kwds={"groups": train_val_df["channel_id"]}
        )
        pred_test = fit.predict(test_df[cols]).values

        sse_test = float(np.sum((y_test - pred_test) ** 2))
        r2_oos_test = 1.0 - (sse_test / sst_test)
        rmse_test = float(np.sqrt(np.mean((y_test - pred_test) ** 2)))
        mae_test = float(np.mean(np.abs(y_test - pred_test)))

        test_df[f"{m_name}_pred"] = pred_test
        test_df[f"{m_name}_err"] = y_test - pred_test

        # Channel level error dispersion
        ch_maes = []
        for ch_name, grp in test_df.groupby("channel_name"):
            c_mae = float(np.mean(np.abs(grp[f"{m_name}_err"])))
            c_rmse = float(np.sqrt(np.mean(grp[f"{m_name}_err"] ** 2)))
            c_bias = float(np.mean(grp[f"{m_name}_err"]))
            ch_maes.append(c_mae)

            test_channel_records.append({
                "model_name": m_name,
                "channel_name": ch_name,
                "subscriber_tier": grp["subscriber_tier"].iloc[0],
                "subscribers": int(grp["subscribers_channel"].iloc[0]),
                "n_videos": len(grp),
                "mae": c_mae,
                "rmse": c_rmse,
                "bias": c_bias
            })

        is_locked = (m_name == "Model_4_Interaction")
        test_summary.append({
            "model_name": m_name,
            "is_locked_model": is_locked,
            "n_trainval_obs": N_trainval,
            "n_trainval_channels": G_trainval,
            "n_test_obs": N_test,
            "n_test_channels": G_test,
            "sse_test": sse_test,
            "sst_test": sst_test,
            "r2_oos_test": r2_oos_test,
            "rmse_test": rmse_test,
            "mae_test": mae_test,
            "mean_channel_mae": float(np.mean(ch_maes)),
            "median_channel_mae": float(np.median(ch_maes)),
            "std_channel_mae": float(np.std(ch_maes)),
            "iqr_channel_mae": float(np.percentile(ch_maes, 75) - np.percentile(ch_maes, 25))
        })

        tag = " [LOCKED SPECIFICATION]" if is_locked else ""
        logger.info(f"{m_name:<25}: R²_OOS = {r2_oos_test:.4f} | RMSE = {rmse_test:.4f} | MAE = {mae_test:.4f}{tag}")

    # Log Refitted Model 4-Centered Coefficients
    fit_m4 = sm.OLS(train_val_df["log_views"], train_val_df[models["Model_4_Interaction"]]).fit(
        cov_type="cluster", cov_kwds={"groups": train_val_df["channel_id"]}
    )
    logger.info("\n" + "=" * 75)
    logger.info("REFITTED MODEL 4-CENTERED ESTIMATES (TRAIN + VAL SAMPLE, G = 32, N = 677)")
    logger.info("=" * 75)
    for var, coef, se, pval in zip(fit_m4.params.index, fit_m4.params, fit_m4.bse, fit_m4.pvalues):
        logger.info(f"  {var:<18}: Coef = {coef:10.4f} | SE = {se:10.4f} | p = {pval:.4f}")

    # Channel Dispersion for Locked Model 4
    df_test_channels = pd.DataFrame(test_channel_records)
    m4_channels = df_test_channels[df_test_channels["model_name"] == "Model_4_Interaction"]

    logger.info("\n" + "=" * 75)
    logger.info("LOCKED MODEL 4-CENTERED TEST CHANNEL-BY-CHANNEL ERROR DISPERSION")
    logger.info("=" * 75)
    for _, r in m4_channels.iterrows():
        logger.info(
            f"  {r['channel_name']:<18} ({r['subscriber_tier']}, {r['subscribers']:,} subs, N={r['n_videos']}): "
            f"MAE = {r['mae']:.4f} | RMSE = {r['rmse']:.4f} | Bias = {r['bias']:+.4f}"
        )

    df_test_summary = pd.DataFrame(test_summary)

    out_test_summary_csv = RESULTS_DIR / "test_set_evaluation.csv"
    out_test_channels_csv = RESULTS_DIR / "test_channel_dispersion.csv"

    df_test_summary.to_csv(out_test_summary_csv, index=False)
    df_test_channels.to_csv(out_test_channels_csv, index=False)

    logger.info("\n" + "=" * 75)
    logger.info(f"Saved test evaluation summary: {out_test_summary_csv}")
    logger.info(f"Saved test channel dispersion: {out_test_channels_csv}")
    logger.info("STEP 6 COMPLETED SUCCESSFULLY: HOLDOUT EVALUATION COMPLETE")
    logger.info("=" * 75)


if __name__ == "__main__":
    run_test_evaluation()
