# Project Progress — Post-Audit Update

**Updated:** 2026-11-10  
**Status:** Post-audit corrections implemented and automated validation rerun.

## Completed in this correction pass
- Fixed phrase/token double counting in title lexicon matching.
- Added five title-matching regression tests (all pass).
- Re-extracted title features and recalculated normalization, weighted/equal/PCA scores.
- Added expression-excluded weighted and equal-weight sensitivity scores.
- Recomputed creator-level and creator-month analysis and regenerated figures.
- Corrected creator-month threshold semantics: all 182 groups are exploratory; maximum cell size is 3.
- Generated face-count disagreement CSV and identified five 480x360 thumbnails.
- Updated research framing, README, methodology, and validation script.
- Ran 8 automated validation checks; all passed.

## Not rerun / still limited
- MediaPipe and FER image processing were not rerun; existing thumbnail features were reused.
- No manual ground-truth adjudication of all face-detection disagreements was performed.
- Predictive validity cannot be assessed without impressions and CTR data.

## Reproduction commands
```bash
python scripts/03_extract_title_features.py
python scripts/05_calculate_scores.py
python scripts/06_creator_analysis.py
python scripts/07_validate_results.py
python tests/test_title_matching.py
```
