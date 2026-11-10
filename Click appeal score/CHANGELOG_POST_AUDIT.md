# Changelog — Post-Audit Corrections (2026-11-10)

- Fixed phrase-first title lexicon matching; phrase spans are excluded from subsequent whitespace-token matches.
- Preserved the original whitespace-token capitalization/token denominator convention.
- Added five regression tests for phrase matching and repeated/standalone tokens.
- Reran title feature extraction, normalization/scoring, creator aggregation/statistical analysis, and automated validation. Thumbnail feature extraction (MediaPipe/FER) was not rerun.
- Added `score_weighted_no_expr` and `score_equal_no_expr` sensitivity variants.
- Added `analysis/sensitivity_analysis.csv` with summary statistics and rank correlations.
- Replaced the invalid creator-month five-video retention proxy with explicit per-cell sample-size status and exploratory interpretation.
- Added `analysis/face_count_disagreements.csv`.
- Updated methodology/README terminology and corrected creator summary values after score recalculation.
- Added post-audit report and expanded automated validation to cover the additional score variants and creator-month sample sizes.
