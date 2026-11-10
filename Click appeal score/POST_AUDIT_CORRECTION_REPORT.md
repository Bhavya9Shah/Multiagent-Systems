# Click Appeal Score — Post-Audit Correction Report

**Correction date:** 2026-11-10  
**Input:** `CLICK_APPEAL_COMPLETE_OUTPUT.zip` supplied in this conversation  
**Scope:** Targeted code, score, creator-analysis, documentation, and validation corrections. No thumbnail download or FER/MediaPipe reprocessing was performed.

## Executive result

The existing project was inspected and corrected in place. The corrected title-feature stage, score calculation, creator analysis, and automated validation were executed successfully. The project contains 186 source rows, 186 thumbnail files, 186 title-feature rows, 186 thumbnail-feature rows, 186 normalized rows, and 186 video-score rows.

## Corrections completed

1. **Phrase/token double-counting:** fixed phrase-first matching so tokens overlapping an already-counted phrase are excluded. Existing whitespace tokenization is preserved. Five regression tests pass. Only two titles changed in the trigger/sensitive counts:
   - `IuqIAmdwXIo` — trigger ratio changed from 0.333333 to 0.166667 (`worth it` counted once, not twice).
   - `5JK_RHpiIRw` — sensitive ratio changed from 0.222222 to 0.111111 (`heart attack` counted once, not twice).
2. **Creator-month threshold:** removed the misleading niche-wide `period_ge_5` proxy. Each creator-month row now reports `meets_5_video_creator_month_minimum` and an explicit interpretation. There are 182 creator-month groups; maximum group size is 3. All are below n=5 and labelled exploratory descriptive only. Creator-level five-video flag is retained.
3. **Capitalization rule:** retained and documented the existing rule: uppercase whitespace tokens with at least two alphabetic letters divided by the whitespace-token count.
4. **Score sensitivity:** added weighted and equal-weight variants excluding the expression feature. This allows comparison where no-face rows are not assigned an additional zero expression component.
5. **Research framing:** revised methodology and README to describe the score as a researcher-defined descriptive composite proxy, not an objective psychological measure or validated click/CTR predictor. The 0.4/0.6 weights remain unchanged and are explicitly researcher-selected.
6. **Face disagreement table:** generated `analysis/face_count_disagreements.csv` with all 75 mismatches. Exact agreement is 111/186; MediaPipe NO_FACE status occurs in 12 rows. Original CSV `face_count` remains preserved and is not treated as ground truth.
7. **Thumbnail resolution:** identified 5 images not at 1280x720: A3L0wiHvrJw (480x360), Ohew6CF7Vuw (480x360), 9XrGxYMu9P8 (480x360), Wq7n9IrcAMY (480x360), UPNSZh2_PmY (480x360). They were not upscaled.
8. **Validation:** extended automated validation to include both expression-excluded score variants and creator-month threshold transparency. All 8 listed automated checks pass. See `analysis/validation_report.csv`.

## Recalculated score summary

| Score | Mean | Min | Max |
|---|---:|---:|---:|
| Title | 0.175348 | 0.000000 | 0.732557 |
| Thumbnail | 0.542824 | 0.072404 | 0.831496 |
| Baseline weighted (0.4/0.6) | 0.395833 | 0.073442 | 0.656941 |
| Equal weight (8 features) | 0.359086 | 0.073702 | 0.645520 |
| PCA alternative | 0.516121 | 0.000000 | 1.000000 |
| Weighted excluding expression | 0.463377 | 0.087923 | 0.735810 |
| Equal weight excluding expression | 0.381083 | 0.084231 | 0.734880 |

Spearman rank correlation between the baseline weighted score and weighted score excluding expression: **0.8832**.

## Creator summary

Top three recalculated creator means: Wake'N'Bite (0.5403), Max vs Food (0.5246), BeardMeatsFood (0.4901). Creator-level aggregation was recomputed from the updated video scores.

## What remains unverified

- MediaPipe face detection and FER probabilities were **not rerun**. Stored predictions were reused.
- Annotated thumbnail files and contact sheets are present in the supplied ZIP (186 annotated thumbnails, plus contact sheets), but the correction process did not manually adjudicate every face box.
- The source and ground-truth validity of the original CSV `face_count` remain unknown.
- The dataset has no impressions/CTR labels; predictive validity and causal effects cannot be established.
- Correlation/regression analyses and PCA are recomputed in the pipeline, but their substantive interpretation remains exploratory.

## Validation interpretation

All automated checks in `analysis/validation_report.csv` passed. This means the listed file-integrity, range, arithmetic, creator aggregation, and sample-size-label checks passed. It does **not** validate FER accuracy, face-detection accuracy, psychological interpretation, or click prediction.
