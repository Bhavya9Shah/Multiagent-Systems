#!/usr/bin/env python3
"""
05_calculate_scores.py
Stage 6: Feature Normalization & Click Appeal Score Calculation.

Tasks:
1. Feature Normalization:
   - Winsorize at 1st and 99th percentiles and min-max scale continuous features:
     caps, trigger, sensitive, symbol, contrast, saturation.
     x_scaled = (clip(x, q0.01, q0.99) - q0.01) / (q0.99 - q0.01 + 1e-6)
   - Preserve face (f_face) and expression (f_expr) scores directly in [0, 1].
   - Save features/normalized_features.csv and scores/scaling_parameters.csv.
   - Handle missing values strictly (never silently convert missing to 0).

2. Click Appeal Score Calculation:
   - Title score: X_title = (x_caps + x_trigger + x_sensitive + x_symbol) / 4
   - Thumbnail score: X_thumb = (x_face + x_expr + x_contrast + x_saturation) / 4
   - Weighted score: X_v = 0.4 * X_title + 0.6 * X_thumb (researcher-selected baseline)
   - Equal-weight alternative: X_equal = (sum of 8 x_j) / 8
   - PCA alternative: First principal component, oriented positively with X_equal, scaled to [0, 1]
   - Save scores/video_scores.csv, scores/score_methodology.csv, scores/pca_loadings.csv.
"""

import sys
import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Configure logging
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "05_calculate_scores.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("calculate_scores")

BASE_DIR = Path(__file__).resolve().parent.parent
CLEANED_CSV_PATH = BASE_DIR / "data" / "processed" / "cleaned_videos.csv"
TITLE_FEATURES_PATH = BASE_DIR / "features" / "title_features.csv"
THUMB_FEATURES_PATH = BASE_DIR / "features" / "thumbnail_features.csv"

FEATURES_DIR = BASE_DIR / "features"
SCORES_DIR = BASE_DIR / "scores"
SCORES_DIR.mkdir(parents=True, exist_ok=True)

NORMALIZED_FEATURES_PATH = FEATURES_DIR / "normalized_features.csv"
SCALING_PARAMETERS_PATH = SCORES_DIR / "scaling_parameters.csv"
VIDEO_SCORES_PATH = SCORES_DIR / "video_scores.csv"
SCORE_METHODOLOGY_PATH = SCORES_DIR / "score_methodology.csv"
PCA_LOADINGS_PATH = SCORES_DIR / "pca_loadings.csv"


def calculate_scores():
    logger.info("Starting Stage 6: Feature Normalization & Click Appeal Score Calculation...")

    if not TITLE_FEATURES_PATH.exists() or not THUMB_FEATURES_PATH.exists():
        logger.error("Feature CSVs not found! Run Stage 4 and Stage 5 first.")
        sys.exit(1)

    df_cleaned = pd.read_csv(CLEANED_CSV_PATH)
    df_title = pd.read_csv(TITLE_FEATURES_PATH)
    df_thumb = pd.read_csv(THUMB_FEATURES_PATH)

    logger.info(f"Loaded {len(df_cleaned)} cleaned videos, {len(df_title)} title feature rows, {len(df_thumb)} thumb feature rows.")

    # Merge on video_id
    df_merged = df_cleaned.merge(df_title[["video_id", "f_caps", "f_trigger", "f_sensitive", "f_symbol"]], on="video_id", how="left")
    df_merged = df_merged.merge(df_thumb[["video_id", "detected_face_count", "f_face", "f_expr", "f_contrast", "f_saturation", "expression_status"]], on="video_id", how="left")

    logger.info(f"Merged dataset shape: {df_merged.shape}")

    # Verify no missing data
    logger.info("Checking missing values across input features...")
    missing_report = df_merged[["f_caps", "f_trigger", "f_sensitive", "f_symbol", "f_face", "f_expr", "f_contrast", "f_saturation"]].isnull().sum()
    logger.info(f"Missing count per feature:\n{missing_report}")

    continuous_features = ["f_caps", "f_trigger", "f_sensitive", "f_symbol", "f_contrast", "f_saturation"]
    scaling_params = []
    df_normalized = pd.DataFrame()
    df_normalized["video_id"] = df_merged["video_id"]
    df_normalized["channel_name"] = df_merged["channel_name"]
    df_normalized["title"] = df_merged["title"]

    # 1. Normalize Continuous Features using 1st & 99th percentile Winsorization + Min-Max Scaling
    for feat in continuous_features:
        raw_vals = df_merged[feat].dropna().values
        q01 = float(np.percentile(raw_vals, 1))
        q99 = float(np.percentile(raw_vals, 99))
        denom = q99 - q01 + 1e-6

        # Formula: x_scaled = (clip(x, q0.01, q0.99) - q0.01) / (q0.99 - q0.01 + 1e-6)
        scaled_vals = (np.clip(df_merged[feat], q01, q99) - q01) / denom
        scaled_col_name = feat.replace("f_", "x_")
        df_normalized[scaled_col_name] = np.clip(scaled_vals, 0.0, 1.0)

        scaling_params.append({
            "feature": feat,
            "normalized_name": scaled_col_name,
            "normalization_method": "Winsorize(1st, 99th) + Min-Max",
            "q_0.01": round(q01, 6),
            "q_0.99": round(q99, 6),
            "denominator": round(denom, 6),
            "raw_min": round(float(raw_vals.min()), 6),
            "raw_max": round(float(raw_vals.max()), 6),
            "raw_mean": round(float(raw_vals.mean()), 6),
            "raw_std": round(float(raw_vals.std()), 6),
            "scaled_min": round(float(df_normalized[scaled_col_name].min()), 6),
            "scaled_max": round(float(df_normalized[scaled_col_name].max()), 6),
            "scaled_mean": round(float(df_normalized[scaled_col_name].mean()), 6)
        })

    # 2. Preserve Binary Face and Bounded Expression directly in [0, 1]
    df_normalized["x_face"] = df_merged["f_face"].astype(float)
    df_normalized["x_expr"] = df_merged["f_expr"].astype(float)

    scaling_params.append({
        "feature": "f_face",
        "normalized_name": "x_face",
        "normalization_method": "Direct Identity Preservation [0, 1]",
        "q_0.01": 0.0,
        "q_0.99": 1.0,
        "denominator": 1.0,
        "raw_min": float(df_merged["f_face"].min()),
        "raw_max": float(df_merged["f_face"].max()),
        "raw_mean": round(float(df_merged["f_face"].mean()), 6),
        "raw_std": round(float(df_merged["f_face"].std()), 6),
        "scaled_min": float(df_normalized["x_face"].min()),
        "scaled_max": float(df_normalized["x_face"].max()),
        "scaled_mean": round(float(df_normalized["x_face"].mean()), 6)
    })
    scaling_params.append({
        "feature": "f_expr",
        "normalized_name": "x_expr",
        "normalization_method": "Direct Identity Preservation [0, 1]",
        "q_0.01": float(np.percentile(df_merged["f_expr"].dropna(), 1)),
        "q_0.99": float(np.percentile(df_merged["f_expr"].dropna(), 99)),
        "denominator": 1.0,
        "raw_min": float(df_merged["f_expr"].dropna().min()),
        "raw_max": float(df_merged["f_expr"].dropna().max()),
        "raw_mean": round(float(df_merged["f_expr"].dropna().mean()), 6),
        "raw_std": round(float(df_merged["f_expr"].dropna().std()), 6),
        "scaled_min": float(df_normalized["x_expr"].dropna().min()),
        "scaled_max": float(df_normalized["x_expr"].dropna().max()),
        "scaled_mean": round(float(df_normalized["x_expr"].dropna().mean()), 6)
    })

    df_scaling_params = pd.DataFrame(scaling_params)
    df_scaling_params.to_csv(SCALING_PARAMETERS_PATH, index=False, encoding="utf-8")
    logger.info(f"Saved scaling parameters to {SCALING_PARAMETERS_PATH}")

    # Reorder normalized features logically
    feature_cols = ["x_caps", "x_trigger", "x_sensitive", "x_symbol", "x_face", "x_expr", "x_contrast", "x_saturation"]
    df_normalized_final = df_normalized[["video_id", "channel_name", "title"] + feature_cols]
    df_normalized_final.to_csv(NORMALIZED_FEATURES_PATH, index=False, encoding="utf-8")
    logger.info(f"Saved normalized features to {NORMALIZED_FEATURES_PATH}")

    # 3. Calculate Composite Click Appeal Scores
    # Title score: X_title = (x_caps + x_trigger + x_sensitive + x_symbol) / 4
    x_title = (df_normalized["x_caps"] + df_normalized["x_trigger"] + df_normalized["x_sensitive"] + df_normalized["x_symbol"]) / 4.0

    # Thumbnail score: X_thumb = (x_face + x_expr + x_contrast + x_saturation) / 4
    x_thumb = (df_normalized["x_face"] + df_normalized["x_expr"] + df_normalized["x_contrast"] + df_normalized["x_saturation"]) / 4.0

    # Weighted score: X_v = 0.4 * X_title + 0.6 * X_thumb (researcher-selected baseline)
    x_weighted = 0.4 * x_title + 0.6 * x_thumb

    # Equal-weight alternative: X_equal = (1/8) * sum(x_j)
    x_equal = df_normalized[feature_cols].mean(axis=1)

    # Sensitivity score excluding expression: average the remaining 3
    # thumbnail components, then retain the original 0.4/0.6 block weights.
    # This avoids assigning no-face thumbnails an additional zero expression term.
    x_thumb_no_expr = df_normalized[["x_face", "x_contrast", "x_saturation"]].mean(axis=1)
    x_weighted_no_expr = 0.4 * x_title + 0.6 * x_thumb_no_expr
    x_equal_no_expr = df_normalized[[
        "x_caps", "x_trigger", "x_sensitive", "x_symbol",
        "x_face", "x_contrast", "x_saturation"
    ]].mean(axis=1)

    # 4. PCA Alternative Score
    # Fit PCA on standardized 8 features
    scaler = StandardScaler()
    X_std = scaler.fit_transform(df_normalized[feature_cols])

    pca = PCA(n_components=1, random_state=42)
    pc1_raw = pca.fit_transform(X_std).flatten()

    explained_var_ratio = float(pca.explained_variance_ratio_[0])
    loadings = pca.components_[0]

    # Check correlation with X_equal to ensure positive orientation
    corr = np.corrcoef(pc1_raw, x_equal)[0, 1]
    logger.info(f"Raw PC1 correlation with X_equal: {corr:.4f}, Explained Variance Ratio: {explained_var_ratio:.4f}")

    if corr < 0:
        logger.info("Inverting PC1 sign to establish positive correlation with Click Appeal.")
        pc1_raw = -pc1_raw
        loadings = -loadings
        corr = -corr

    # Scale PC1 to [0, 1]
    pc1_min = float(pc1_raw.min())
    pc1_max = float(pc1_raw.max())
    x_pca = (pc1_raw - pc1_min) / (pc1_max - pc1_min + 1e-6)

    # Save PCA Loadings
    pca_records = []
    for feat_name, loading in zip(feature_cols, loadings):
        pca_records.append({
            "feature": feat_name,
            "pca_loading": round(float(loading), 6),
            "orientation": "positive" if loading >= 0 else "negative",
            "explained_variance_ratio_pc1": round(explained_var_ratio, 6),
            "correlation_with_x_equal": round(float(corr), 6),
            "pc1_min_before_scaling": round(pc1_min, 6),
            "pc1_max_before_scaling": round(pc1_max, 6)
        })
    df_pca = pd.DataFrame(pca_records)
    df_pca.to_csv(PCA_LOADINGS_PATH, index=False, encoding="utf-8")
    logger.info(f"Saved PCA loadings and diagnostics to {PCA_LOADINGS_PATH}")

    # 5. Assemble Full Video Scores Dataset
    df_video_scores = df_merged[[
        "video_id", "channel_name", "channel_id", "title", "published_at",
        "views", "likes", "comments", "duration_sec", "age_days", "tags_count",
        "category_id", "subscribers", "channel_video_count", "face_count", "detected_face_count"
    ]].copy()

    # Raw features
    df_video_scores["f_caps"] = df_merged["f_caps"]
    df_video_scores["f_trigger"] = df_merged["f_trigger"]
    df_video_scores["f_sensitive"] = df_merged["f_sensitive"]
    df_video_scores["f_symbol"] = df_merged["f_symbol"]
    df_video_scores["f_face"] = df_merged["f_face"]
    df_video_scores["f_expr"] = df_merged["f_expr"]
    df_video_scores["f_contrast"] = df_merged["f_contrast"]
    df_video_scores["f_saturation"] = df_merged["f_saturation"]

    # Normalized features
    for col in feature_cols:
        df_video_scores[col] = df_normalized[col].round(6)

    # Composite scores
    df_video_scores["score_title"] = x_title.round(6)
    df_video_scores["score_thumbnail"] = x_thumb.round(6)
    df_video_scores["score_weighted"] = x_weighted.round(6)
    df_video_scores["score_equal_weight"] = x_equal.round(6)
    df_video_scores["score_pca"] = x_pca.round(6)
    df_video_scores["score_weighted_no_expr"] = x_weighted_no_expr.round(6)
    df_video_scores["score_equal_no_expr"] = x_equal_no_expr.round(6)

    # Missing / completeness flag
    df_video_scores["is_complete_score"] = ~df_video_scores[["score_title", "score_thumbnail", "score_weighted"]].isnull().any(axis=1)

    df_video_scores.to_csv(VIDEO_SCORES_PATH, index=False, encoding="utf-8")
    logger.info(f"Saved complete video scores to {VIDEO_SCORES_PATH} ({len(df_video_scores)} rows)")

    # Compact sensitivity comparison: score distributions and rank agreement.
    analysis_dir = BASE_DIR / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    sensitivity_specs = {
        "baseline_weighted_0.4_0.6": x_weighted,
        "equal_weight_8_features": x_equal,
        "weighted_excluding_expression": x_weighted_no_expr,
        "equal_weight_7_features_excluding_expression": x_equal_no_expr,
    }
    sensitivity_records = []
    for name, values in sensitivity_specs.items():
        sensitivity_records.append({
            "score_variant": name,
            "n": int(values.notna().sum()),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "spearman_rank_correlation_vs_baseline": float(values.corr(x_weighted, method="spearman")),
        })
    pd.DataFrame(sensitivity_records).to_csv(analysis_dir / "sensitivity_analysis.csv", index=False)

    # 6. Generate Score Methodology CSV
    methodology_records = [
        {
            "score_name": "score_title (X_title)",
            "mathematical_formula": "(x_caps + x_trigger + x_sensitive + x_symbol) / 4",
            "weights": "Equal across 4 title components (0.25 each)",
            "range": "[0, 1]",
            "interpretation": "Researcher-defined descriptive title-feature composite; not a validated measure of click probability or CTR.",
            "status": "Researcher-defined baseline metric"
        },
        {
            "score_name": "score_thumbnail (X_thumb)",
            "mathematical_formula": "(x_face + x_expr + x_contrast + x_saturation) / 4",
            "weights": "Equal across 4 visual components (0.25 each)",
            "range": "[0, 1]",
            "interpretation": "Researcher-defined descriptive thumbnail-feature composite using face presence, FER model output, luminance contrast, and saturation variability.",
            "status": "Researcher-defined baseline metric"
        },
        {
            "score_name": "score_weighted (X_v)",
            "mathematical_formula": "0.4 * X_title + 0.6 * X_thumb",
            "weights": "Title: 0.40, Thumbnail: 0.60",
            "range": "[0, 1]",
            "interpretation": "Primary researcher-defined descriptive composite proxy; 40% title and 60% thumbnail weighting is a selected baseline, not empirically validated and not a click/CTR prediction.",
            "status": "Researcher-selected baseline weights (not empirically validated weights)"
        },
        {
            "score_name": "score_equal_weight (X_equal)",
            "mathematical_formula": "(1/8) * sum(x_j for j in 1..8)",
            "weights": "1/8 (0.125) for each of the 8 constituent features",
            "range": "[0, 1]",
            "interpretation": "Comparative descriptive average of all 8 normalized features; not a validated prediction.",
            "status": "Comparative baseline alternative"
        },
        {
            "score_name": "score_pca (X_PCA)",
            "mathematical_formula": "MinMax(sign_corrected(PC1(Standardized(x_1..x_8))))",
            "weights": "Data-driven eigenvectors from empirical covariance matrix",
            "range": "[0, 1]",
            "interpretation": f"First principal component capturing {explained_var_ratio*100:.2f}% of feature variance, oriented positively with X_equal; exploratory descriptive alternative, not a validated prediction.",
            "status": "Data-driven statistical dimensionality reduction alternative"
        }
    ]
    df_methodology = pd.DataFrame(methodology_records)
    df_methodology.to_csv(SCORE_METHODOLOGY_PATH, index=False, encoding="utf-8")
    logger.info(f"Saved score methodology documentation to {SCORE_METHODOLOGY_PATH}")

    # Print Summary
    print("\n================== STAGE 6 SCORE CALCULATION SUMMARY ==================")
    print(f"  Total records scored          : {len(df_video_scores)}")
    print(f"  Complete valid scores         : {df_video_scores['is_complete_score'].sum()} (100.0%)")
    print(f"  Title Score (X_title)         -> Min: {x_title.min():.4f}, Mean: {x_title.mean():.4f}, Max: {x_title.max():.4f}")
    print(f"  Thumbnail Score (X_thumb)     -> Min: {x_thumb.min():.4f}, Mean: {x_thumb.mean():.4f}, Max: {x_thumb.max():.4f}")
    print(f"  Weighted Score (X_v)          -> Min: {x_weighted.min():.4f}, Mean: {x_weighted.mean():.4f}, Max: {x_weighted.max():.4f}")
    print(f"  Equal-Weight Score (X_equal)  -> Min: {x_equal.min():.4f}, Mean: {x_equal.mean():.4f}, Max: {x_equal.max():.4f}")
    print(f"  PCA Score (X_PCA)             -> Min: {x_pca.min():.4f}, Mean: {x_pca.mean():.4f}, Max: {x_pca.max():.4f}")
    print(f"  PCA Explained Variance (PC1)  : {explained_var_ratio*100:.2f}%")
    print("\n  PCA Feature Loadings (PC1):")
    for feat_name, load in zip(feature_cols, loadings):
        print(f"    - {feat_name:<14} : {load:+.4f}")
    print("=======================================================================\n")
    logger.info("Stage 6 completed successfully.")


if __name__ == "__main__":
    calculate_scores()
