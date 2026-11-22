# Click Appeal Score

A reproducible Python pipeline that constructs a **researcher-defined descriptive composite proxy** from observable YouTube video title and thumbnail characteristics for **809 unique videos** across 38 creator channels.

This directory contains the finalized, self-contained Click Appeal Score component of our Multi-Agent Systems research project (*Click Appeal as a Potential Game: Modeling Creator Incentives, Attention Congestion, and Social Welfare in YouTube*).

---

## 1. What the Pipeline Does

1. **Cleans & Merges Datasets:** Integrates the raw benchmark files from `data/original/`, recovers all 12 corrupted YouTube video IDs (`#NAME?`) from high-resolution thumbnail URLs, repairs channel anomalies, and verifies 0 duplicate IDs across all 809 videos.
2. **Retrieves Thumbnails:** Verifies and stores all 809 original JPEG thumbnails in `thumbnails_original/` and generates `thumbnails_original/download_manifest.csv`.
3. **Extracts Title Features:** Computes capitalization ratio ($f_{\text{caps}}$), trigger-word ratio ($f_{\text{trigger}}$), sensitive-word ratio ($f_{\text{sensitive}}$), and symbol/emoji ratio ($f_{\text{symbol}}$).
4. **Extracts Thumbnail Features:** Uses MediaPipe Face Detection (`model_selection=1, min_detection_confidence=0.5`) to identify face presence ($f_{\text{face}}$), FER 25.10.3 to classify emotion intensity ($f_{\text{expr}} = \max(P_{\text{surprise}}, P_{\text{fear}})$) on the largest face, and computes luminance contrast ($f_{\text{contrast}}$) and HSV saturation ($f_{\text{saturation}}$).
5. **Normalizes Features:** Uses established 1st and 99th percentile winsorization followed by min-max scaling for continuous features, preserving face and expression directly in $[0, 1]$.
6. **Calculates Click Appeal Scores:** Computes title appeal, thumbnail appeal, primary weighted Click Appeal Score ($X_v$), equal-weight alternative, and PCA alternative.
7. **Aggregates Creator Metrics:** Aggregates scores across all 38 creator channels, computing leave-one-creator-out (LOCO) peer benchmarks and monthly dynamics.
8. **Validates Results:** Verifies row accountability, mathematical reconstructibility, score boundedness, and generates publication-quality figures.

---

## 2. Core Score Definitions

Let the normalized title features be $x_{\text{caps}}$, $x_{\text{trigger}}$, $x_{\text{sensitive}}$, and $x_{\text{symbol}}$, and the normalized thumbnail features be $x_{\text{face}}$, $x_{\text{expr}}$, $x_{\text{contrast}}$, and $x_{\text{saturation}}$ (all in $[0, 1]$):

- **Title Click Appeal Score:** $X_{\text{title}} = \frac{x_{\text{caps}} + x_{\text{trigger}} + x_{\text{sensitive}} + x_{\text{symbol}}}{4}$
- **Thumbnail Click Appeal Score:** $X_{\text{thumb}} = \frac{x_{\text{face}} + x_{\text{expr}} + x_{\text{contrast}} + x_{\text{saturation}}}{4}$
- **Primary Weighted Click Appeal Score ($X_v$):** $X_v = 0.4 X_{\text{title}} + 0.6 X_{\text{thumb}}$
- **Equal-Weight Alternative:** $X_{\text{equal}} = \frac{1}{8} \sum_{j=1}^8 x_j$

---

## 3. Directory Layout & Key Deliverables

```text
Click appeal score/
│
├── data/
│   ├── original/          # Raw input video metadata CSVs
│   └── processed/
│       ├── cleaned_videos.csv    # Final working dataset of 809 unique videos
│       ├── cleaning_log.csv      # Audit trail of the 12 recovered #NAME? IDs
│       └── download_manifest.csv # Download status & dimensions for all 809 thumbnails
│
├── thumbnails_original/   # 809 original JPEG thumbnails
│   └── download_manifest.csv
│
├── annotated_thumbnails/  # 809 visual audit overlays with face bounding boxes & emotions
│   └── contact_sheet.jpg  # 24-image representative visual audit contact sheet
│
├── features/
│   ├── title_features.csv      # 4 title lexical features for 809 videos
│   ├── thumbnail_features.csv  # 4 vision features (MediaPipe + FER + Contrast + Saturation)
│   └── normalized_features.csv # Winsorized & min-max scaled features in [0, 1]
│
├── scores/
│   ├── video_scores.csv           # Final video-level Click Appeal Scores (N=809)
│   ├── creator_summary_scores.csv # Channel-level descriptive stats & LOCO benchmarks (38 channels)
│   ├── creator_monthly_scores.csv # Monthly aggregated scores by channel
│   ├── scaling_parameters.csv     # Established 1st/99th percentiles & denominators
│   ├── pca_loadings.csv           # Baseline PCA eigenvector loadings
│   └── score_methodology.csv      # Formal formulas and definitions
│
├── analysis/
│   ├── 01_score_distribution.png         # Distribution histogram & KDE for the 809 videos
│   ├── 02_channel_appeal_comparison.png   # Ranked mean Click Appeal Scores across 38 channels
│   ├── 03_score_components_breakdown.png # Title score vs. Thumbnail score scatter & trend
│   ├── contact_sheet.jpg                 # Visual audit contact sheet
│   ├── validation_report.csv             # Automated validation check results
│   └── VALIDATION_REPORT.md              # Quality assurance & descriptive summary report
│
├── scripts/
│   ├── 01_clean_data.py               # Stage 2: Data cleaning, recovery, and superset merging
│   ├── 02_download_thumbnails.py       # Stage 3: Thumbnail retrieval and verification
│   ├── 03_extract_title_features.py    # Stage 4: Lexical feature extraction
│   ├── 04_extract_thumbnail_features.py# Stage 5: MediaPipe face & FER emotion detection
│   ├── 05_calculate_scores.py          # Stage 6: Normalization & score calculation
│   ├── 06_creator_analysis.py          # Stage 7: Channel aggregation & LOCO benchmarks
│   └── 07_validate_results.py          # Stage 8: Mathematical & data integrity validation
│
├── logs/                  # Detailed execution logs for each pipeline stage
├── run_pipeline.py        # Master orchestrator script (runs all stages in sequence)
├── requirements.txt       # Environment dependencies
└── README.md              # Pipeline documentation
```

---

## 4. Reproducibility & How to Run

### Complete End-to-End Execution
To rerun the full pipeline from scratch:

```bash
python run_pipeline.py
```

### Running Individual Stages
Each stage can also be executed independently:

```bash
python scripts/01_clean_data.py
python scripts/02_download_thumbnails.py
python scripts/03_extract_title_features.py
python scripts/04_extract_thumbnail_features.py
python scripts/05_calculate_scores.py
python scripts/06_creator_analysis.py
python scripts/07_validate_results.py
```

---

## 5. Summary Statistics ($N = 809$ Videos)

| Metric | Formula / Role | Mean | Std | Min | Median | Max |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`score_weighted` ($X_v$)** | **$0.4 X_{\text{title}} + 0.6 X_{\text{thumb}}$ (Primary)** | **0.3924** | **0.1037** | **0.0054** | **0.3977** | **0.6843** |
| `score_title` | $\frac{x_{\text{caps}} + x_{\text{trig}} + x_{\text{sens}} + x_{\text{sym}}}{4}$ | 0.1659 | 0.1395 | 0.0000 | 0.1250 | 0.7326 |
| `score_thumbnail` | $\frac{x_{\text{face}} + x_{\text{expr}} + x_{\text{cont}} + x_{\text{sat}}}{4}$ | 0.5434 | 0.1377 | 0.0091 | 0.5512 | 0.8898 |
| `score_equal_weight` | $\frac{1}{8} \sum_{j=1}^8 x_j$ | 0.3546 | 0.1023 | 0.0045 | 0.3551 | 0.6620 |
| `score_pca` | Projection onto baseline PC1 | 0.5340 | 0.1681 | 0.0000 | 0.5367 | 1.0000 |

All 809 videos have valid scores strictly bounded within $[0, 1]$ with zero missing data.
