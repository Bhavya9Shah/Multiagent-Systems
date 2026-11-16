# Empirical Payoff Function Estimation & Econometric Analysis

**CS 329: Foundations of AI: Multi-Agent Systems (Autumn 2026)**  
*Indian Institute of Technology Gandhinagar*  
**Project:** Click Appeal as a Potential Game: Modeling Creator Incentives, Attention Congestion, and Social Welfare in YouTube  
**Author:** Bhavya Shah (Roll: 24110324) — *Payoff Function Estimation, Econometric Regression Analysis & Statistical Inference*  
**Collaborators:**  
- Ankita Kushwaha (24110043) — *Multimodal Scoring & Data Preprocessing*  
- Khushi Shah (24110326) — *Game-Theoretic Formulation & Social Welfare Analysis*  
- Harshita Singh (24110342) — *Platform Mediation & Multi-Agent Simulation*  

---

## 1. Overview & Purpose

This component provides the empirical foundation for modeling YouTube creator packaging competition as an **Exact Potential Game**. Rather than assuming synthetic or arbitrary payoff functions, we estimate the structural parameters of the creator revenue function directly from multimodal video packaging metrics and YouTube performance data ($N = 809$ videos across 38 distinct creator channels).

The fitted parameters, cluster-robust standard errors, and covariance matrices are delivered directly to the game theory component (Khushi Shah) for pure-strategy Nash Equilibrium derivation, best-response dynamics, and Price of Anarchy analysis.

---

## 2. Methodological Architecture (8-Step Pipeline)

Our empirical pipeline follows a pre-registered, leak-free design across 8 sequential steps:

1. **Step 1: Data Ingestion & Integrity Validation:** Ingests `video_scores.csv` and `cleaned_videos.csv`, validates exact 1:1 join ($N = 809$), standardizes subscriber counts to channel-level medians, and verifies zero non-positive views.
2. **Step 2: Channel-Level Stratified Partitioning:** Partitions the 38 channels into 26 Training channels ($N = 541$), 6 Validation channels ($N = 136$), and 6 Held-Out Test channels ($N = 132$), preserving exact subscriber-tier distributions.
3. **Step 3: Primary Model Development (Training Sample):** Fits progressive OLS models (M0 through M5) on the 26 training channels. Applies mean-centering to eliminate structural multicollinearity (VIFs drop from $1{,}334.9$ to $1.15$).
4. **Step 4: 26-Fold LOCO-CV on Training Cohort:** Evaluates predictive generalization via strict Leave-One-Channel-Out Cross-Validation within the training cohort with zero data leakage.
5. **Step 5: Validation Evaluation & Specification Freezing:** Evaluates candidate models on the 6 validation channels ($N = 136$) and formally locks **Model 4-Centered (Strategic Payoff Function)**.
6. **Step 6: Final Holdout Test Evaluation:** Evaluates the locked Model 4 specification on the untouched 6 test channels ($N = 132$) after refitting on the pooled 32 training+validation channels ($R^2_{\mathrm{OOS}} = 0.7537$, $\mathrm{RMSE} = 1.1382$).
7. **Step 7: Post-Test Full-Sample Structural Calibration:** Estimates Model 4-Centered on all 38 channels ($N = 809$) for structural parameter handoff, executes Null-Restricted Wild Cluster Bootstrap tests ($B = 2{,}000$), computes unrestricted bootstrap confidence intervals, and audits the contraction mapping condition ($\rho = 1.2683$).
8. **Step 8: Publication Visualizations:** Generates 6 high-resolution (300 DPI) publication-quality econometric figures in `plots/`.

---

## 3. Core Structural Specification (Model 4-Centered)

The primary econometric specification for creator $i$ on video $t$ is:

$$\ln(V_{it}) = \beta_0^* + \beta_1^* (x_{it} - \mu_x) + \beta_2^* (x_{it} - \mu_x)^2 + \beta_3^* (\bar{x}_{-i,t} - \mu_{\bar{x}}) + \beta_5^* (x_{it} - \mu_x)(\bar{x}_{-i,t} - \mu_{\bar{x}}) + \gamma_s \ln(s_i) + \gamma_d \ln(\text{dur}_{it}) + \gamma_a \text{age}_{it} + \varepsilon_{it}$$

- $x_{it} \in [0, 1]$: Multimodal click appeal score of video $t$ by creator $i$.
- $\bar{x}_{-i,t} \in [0, 1]$: Competitor packaging average within a rolling 60-day window $[t - 30\text{d}, t + 30\text{d}]$, excluding creator $i$.
- $\mu_x, \mu_{\bar{x}}$: Sample centering means ($\mu_x = 0.3924, \mu_{\bar{x}} = 0.3914$).
- $s_i$: Creator subscriber count (fixed scale control).
- $\text{dur}_{it}$: Video duration in seconds.
- $\text{age}_{it}$: Elapsed days since publication.

---

## 4. Key Empirical & Theoretical Findings

### Full-Sample Calibrated Coefficients ($N = 809, G = 38$)
- $\hat{\beta}_1^* = -0.3135$ ($\mathrm{SE}_{\mathrm{CRSE}} = 1.9191, p = 0.8711$)
- $\hat{\beta}_2^* = -8.1974$ ($\mathrm{SE}_{\mathrm{CRSE}} = 8.2994, p = 0.3297$) — *Point estimate is strictly concave ($\beta_2 < 0$), reflecting diminishing returns.*
- $\hat{\beta}_3^* = +2.6172$ ($\mathrm{SE}_{\mathrm{CRSE}} = 6.0386, p = 0.6672$)
- $\hat{\beta}_5^* = +20.7942$ ($\mathrm{SE}_{\mathrm{CRSE}} = 60.2260, p = 0.7318$)
- $\hat{\gamma}_s = +1.0417$ ($\mathrm{SE}_{\mathrm{CRSE}} = 0.1147, p < 0.0001$) — *Creator scale elasticity is the dominant predictor ($t = 9.08$).*

### Statistical Inference via Wild Cluster Bootstrap ($B = 2{,}000$)
- **Diminishing Returns ($H_0: \beta_2 \ge 0$):** One-sided WCB $p = 0.2480$. Retained at $\alpha = 0.05$. Concavity is a calibrated theoretical feature rather than a statistically confirmed finding.
- **Strategic Interaction ($H_0: \beta_5 = 0$):** Two-sided WCB $p = 0.7835$. Retained at $\alpha = 0.05$.
- **Contraction Ratio ($\rho = |\beta_5| / (2|\beta_2|)$):** Point estimate $\rho = 1.2683$. $100\%$ of bootstrap draws are concave ($\beta_2 < 0$), with $P(\rho < 1) = 32.40\%$.

### Handoff Condition for Game Theory (Khushi Shah)
Because $\rho = 1.2683 > 1$ under empirical revenue alone, creators face boundary solutions without convex costs. Incorporating quadratic production costs $C_i(x_i) = c_i x_i + \frac{\kappa_i}{2} x_i^2$ with:
$$\kappa_{\min} > |\beta_5^*| - 2 |\beta_2^*| = 20.7942 - 16.3948 = 4.40$$
guarantees that creator best-response dynamics form a strict contraction mapping on $[0, 1]$, proving the existence and uniqueness of a pure-strategy Nash Equilibrium.

---

## 5. Directory Structure & Key Artifacts

```text
payoff_estimation/
├── data/
│   ├── processed_payoff_data.csv       # Cleaned, joined dataset (N=809)
│   ├── data_dictionary.csv             # 21 variables documented with ranges
│   ├── channel_splits.json             # Stratified channel partition map (seed=42)
│   └── partitioned_payoff_data.csv     # Full dataset labeled with partition
├── scripts/
│   ├── 01_prepare_payoff_data.py       # Step 1: Ingestion & audit
│   ├── 02_create_stratified_split.py   # Step 2: Stratified split (26/6/6)
│   ├── 03_train_models.py              # Step 3: Primary regressions & robustness
│   ├── 04_evaluate_loco_cv.py          # Step 4: 26-fold LOCO-CV on training
│   ├── 05_validate_and_freeze.py       # Step 5: Validation & model locking
│   ├── 06_evaluate_test_holdout.py     # Step 6: Holdout test evaluation
│   ├── 07_post_test_structural_calibration.py # Step 7: Full estimation & bootstrap
│   ├── 08_generate_plots.py            # Step 8: Publication plots generator
│   └── 09_backward_looking_robustness.py # Secondary peer window robustness
├── results/
│   ├── game_theory_coefficients.json   # Structural handoff payload for Game Theory
│   ├── game_theory_coefficients.csv    # Flat parameter summary table
│   ├── post_test_structural_regression_results.csv # Full-sample regression table
│   ├── bootstrap_inference_results.csv # Bootstrap CIs and contraction stats
│   ├── test_set_evaluation.csv         # Holdout test set metrics
│   ├── test_channel_dispersion.csv     # Test channel-by-channel breakdown
│   ├── validation_evaluation.csv       # Validation cohort metrics
│   ├── validation_channel_dispersion.csv # Validation channel breakdown
│   ├── loco_cv_aggregate_summary.csv   # 26-fold LOCO-CV aggregate summary
│   ├── training_model_comparison.csv   # Training fit, AIC, and BIC
│   ├── training_multicollinearity_diagnostics.csv # VIF comparison table
│   ├── subscriber_robustness_table.csv # Scale and FE robustness specifications
│   └── backward_looking_peer_robustness.csv # Contemporaneous vs backward peer table
├── plots/
│   ├── 01_model_progression_fit.png    # In-sample vs OOS R² & AIC/BIC curves
│   ├── 02_actual_vs_predicted.png      # Actual vs predicted for Train/Val/Test
│   ├── 03_residual_audit.png           # 4-panel residual diagnostic grid
│   ├── 04_payoff_response_curves.png   # Revenue response vs convex cost tradeoffs
│   ├── 05_peer_interaction_heatmap.png # 2D contour surface with best-response path
│   └── 06_channel_error_dispersion.png # Channel-level MAE and bias across 38 channels
├── FINAL_PAYOFF_REPORT.md              # Comprehensive master research report
└── README.md                           # This document
```

---

## 6. Reproduction Instructions

All analyses run reproducibly under Python 3.10+ (requires `statsmodels`, `scipy`, `pandas`, `numpy`, and `matplotlib`).

To execute the entire empirical pipeline from Step 1 to Step 8:

```bash
cd payoff_estimation

# Step 1: Data Preparation & Verification
python scripts/01_prepare_payoff_data.py

# Step 2: Channel-Level Stratified Partitioning (26 train, 6 val, 6 test)
python scripts/02_create_stratified_split.py

# Step 3: Model Development & Multicollinearity Diagnostics (Training Cohort)
python scripts/03_train_models.py

# Step 4: 26-Fold LOCO-CV Evaluation (Training Cohort)
python scripts/04_evaluate_loco_cv.py

# Step 5: Model Selection & Specification Freezing (Validation Cohort)
python scripts/05_validate_and_freeze.py

# Step 6: Final Holdout Evaluation (6 Test Channels)
python scripts/06_evaluate_test_holdout.py

# Step 7: Post-Test Full-Sample Structural Calibration & Bootstrap Inference
python scripts/07_post_test_structural_calibration.py

# Step 8: Publication Figure Generation
python scripts/08_generate_plots.py

# Secondary Robustness Check: Backward-Looking Peer Window
python scripts/09_backward_looking_robustness.py
```

*Execution of the entire pipeline completes in approximately 60 seconds.*
