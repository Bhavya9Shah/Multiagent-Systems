# Click Appeal Score

A reproducible Python pipeline that constructs a **researcher-defined descriptive composite proxy** from selected YouTube video title and thumbnail characteristics.

This repository contains only the Click Appeal Score component of the larger group project. Payoff regression, Nash equilibrium, social optimum, Price of Anarchy, platform mediation, and learning simulations are outside this repository's scope and are handled separately by the group.

## What the pipeline does

1. Cleans the supplied video metadata CSV while preserving all source rows.
2. Downloads thumbnails from the URLs in the CSV (existing valid local files are reused).
3. Extracts four title features: capitalization, trigger terms, sensitive terms, and punctuation/emoji.
4. Extracts four thumbnail features: face presence, FER surprise/fear prediction for the largest detected face, luminance contrast, and saturation variability.
5. Normalizes selected continuous features using 1st/99th percentile winsorization followed by min-max scaling.
6. Calculates title, thumbnail, weighted, equal-weight, and PCA alternative scores.
7. Produces creator-level and exploratory creator-month summaries, analysis figures, and validation checks.

## Main score definition

Let the normalized title features be `x_caps`, `x_trigger`, `x_sensitive`, and `x_symbol`, and the normalized thumbnail features be `x_face`, `x_expr`, `x_contrast`, and `x_saturation`.

- `X_title = mean(x_caps, x_trigger, x_sensitive, x_symbol)`
- `X_thumbnail = mean(x_face, x_expr, x_contrast, x_saturation)`
- `X_weighted = 0.4 * X_title + 0.6 * X_thumbnail`
- `X_equal = mean(all eight normalized features)`

The 0.4/0.6 weighting is a researcher-selected baseline, not an empirically validated weighting scheme. See `documentation/CLICK_APPEAL_SCORE_METHODOLOGY_AND_VERIFICATION.md` for operational definitions, normalization details, and limitations.

## Repository contents

```text
.
├── data/
│   ├── original/          # Original input CSV; do not edit
│   └── processed/         # Cleaned data, audit logs, quality report
├── thumbnails_original/  # Downloaded source thumbnails and manifest
├── annotated_thumbnails/ # Stored visual audit images/contact sheets
├── features/              # Raw and normalized feature tables
├── scores/                # Video scores, scaling parameters, creator summaries
├── analysis/              # Validation tables, analysis CSVs and figures
├── scripts/               # Individual pipeline stages
├── tests/                 # Title lexicon matching regression tests
├── documentation/         # Methodology and lexicon documentation
├── requirements.txt
├── run_pipeline.py
└── README.md
```

## Reproducibility: Windows setup

Use 64-bit Python 3.12. Open PowerShell in the repository root (the folder containing `run_pipeline.py`).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the complete pipeline:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py
```

The pipeline runs the scripts in order and stops if a stage fails. It writes execution logs to `logs/`.

### Run individual stages

```powershell
.\.venv\Scripts\python.exe scripts/01_clean_data.py
.\.venv\Scripts\python.exe scripts/02_download_thumbnails.py
.\.venv\Scripts\python.exe scripts/03_extract_title_features.py
.\.venv\Scripts\python.exe scripts/04_extract_thumbnail_features.py
.\.venv\Scripts\python.exe scripts/05_calculate_scores.py
.\.venv\Scripts\python.exe scripts/06_creator_analysis.py
.\.venv\Scripts\python.exe scripts/07_validate_results.py
```

If copying commands, note that the thumbnail-feature stage requires the MediaPipe and TensorFlow/FER dependencies and can take longer than the other stages. The downloader requires internet access if thumbnails are not already present. The pipeline does not require a YouTube API key because it uses the thumbnail URLs already stored in the supplied CSV.

## Tests

Run the title phrase/token overlap regression tests:

```powershell
.\.venv\Scripts\python.exe tests/test_title_matching.py
```

Expected final line: `5 title matching regression tests: PASS`.

## Important interpretation limits

- This score is a descriptive heuristic proxy built from selected observable title/thumbnail cues. It is not an objective psychological measurement or a validated click-probability model.
- Views are not impressions or click-through rate (CTR). Without impressions and CTR data, predictive validity for clicks cannot be established.
- FER values are model predictions, not direct measurements of human emotion.
- Face detection can miss faces or produce false detections. The original CSV `face_count` is preserved for comparison and is not treated as ground truth.
- The trigger/sensitive lexicons are researcher-defined and may contain false positives or miss relevant language.
- Creator-month groups have small sample sizes and are exploratory descriptive summaries, not reliable inferential estimates.
- The dataset contains a limited sample of creators/videos; generalisation beyond this sample is not established.

## Current dataset snapshot

The supplied source contains 186 video rows and 38 channel IDs after cleaning. Four malformed video IDs in the raw CSV were recovered from their thumbnail URLs. Five downloaded thumbnails have resolution 480x360 rather than 1280x720; they are retained at their original resolution and identified in the audit report.

## Audit and change history

- `POST_AUDIT_CORRECTION_REPORT.md` records the independent audit findings and corrections made.
- `CHANGELOG_POST_AUDIT.md` records code/documentation corrections.
- `analysis/validation_report.csv` contains the automated validation results from the supplied run. Re-run the pipeline to independently reproduce them on your machine.
- `PROGRESS.md` records pipeline status and limitations.

## GitHub note

Commit the extracted repository contents, **not** the ZIP archive. The individual thumbnail files are below GitHub's 100 MB per-file limit. Keep the repository private until the group confirms whether the dataset and thumbnail images may be shared publicly. Do not commit API keys, passwords, or personal credentials.
