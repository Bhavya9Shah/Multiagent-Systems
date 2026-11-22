# Empirical Payoff Function Estimation & Econometric Analysis

## 1. Overview & Purpose

This component provides the empirical foundation for modeling YouTube creator packaging competition as a **Continuous Aggregative Exact Potential Game**. Rather than assuming synthetic or arbitrary payoff functions, we estimate the structural parameters of the creator revenue function directly from multimodal video packaging metrics and YouTube performance data ($N = 809$ videos across 38 distinct creator channels).

The fitted parameters, delta-method covariance matrices, and well-posedness conditions are delivered directly to the game theory component for pure-strategy Nash Equilibrium derivation, best-response dynamics, and Price of Anarchy analysis.

> [!NOTE]
> **Data vs. Proposal Scope Clarification:**  
> The dataset covers 38 channels from **global English-language digital entertainment** (Gaming, Comedy, General Entertainment, Food/Tech Lifestyle) rather than the Indian tech-review market proposed in the initial proposal. The analyzed sample comprises 809 videos across creators spanning from $118\text{K}$ to $526\text{M}$ subscribers, capturing real-world cross-channel competition across global attention categories.

---

## 2. Methodological Architecture (10-Stage Pipeline)

Our empirical pipeline follows a pre-registered, leak-free design across 10 sequential stages:

1. **Stage 1 (`01_prepare_payoff_data.py`): Data Ingestion & Audit:** Merges multimodal scores (`video_scores.csv`) with video metadata (`cleaned_videos.csv`), validates 1:1 join ($N = 809$), standardizes subscriber counts to channel-level medians, and verifies zero non-positive views.
2. **Stage 2 (`02_create_stratified_split.py`): Channel-Stratified Splitting:** Partitions the 38 channels into 26 Training channels ($N = 541$), 6 Validation channels ($N = 136$), and 6 Held-Out Test channels ($N = 132$), preserving exact subscriber-tier distributions.
3. **Stage 3 (`03_train_models.py`): Primary Model Development (Training Sample):** Fits progressive OLS models (M0 through M5) on the 26 training channels. Applies mean-centering to eliminate structural multicollinearity (VIFs drop from $1{,}334.9$ to $1.15$).
4. **Stage 4 (`04_evaluate_loco_cv.py`): 26-Fold LOCO-CV on Training Cohort:** Evaluates predictive generalization via strict Leave-One-Channel-Out Cross-Validation within the training cohort with zero data leakage.
5. **Stage 5 (`05_validate_and_freeze.py`): Validation Evaluation & Specification Freezing:** Evaluates candidate models on the 6 validation channels ($N = 136$) and formally locks **Model 4-Centered (Strategic Payoff Function)** based on theoretical necessity.
6. **Stage 6 (`06_evaluate_test_holdout.py`): Final Holdout Test Evaluation:** Evaluates the locked Model 4 specification on the internal holdout of 6 test channels ($N = 132$) after refitting on the pooled 32 training+validation channels ($R^2_{\mathrm{OOS}} = 0.7537$, $\mathrm{RMSE} = 1.1382$).
7. **Stage 7 (`07_post_test_structural_calibration.py`): Full Calibration & Bootstrap Inference:** Estimates Model 4-Centered on all 38 channels ($N = 809$), derives uncentered parameters analytically for paper Eq. 2, executes Wild Cluster Bootstrap tests ($B = 2{,}000$), and computes cluster-pairs bootstrap confidence intervals.
8. **Stage 8 (`08_generate_plots.py`): Publication Figures:** Generates 6 high-resolution (300 DPI) publication-quality econometric figures in `plots/`.
9. **Stage 9 (`09_backward_looking_robustness.py`): Backward-Looking Peer Window Robustness:** Compares the primary contemporaneous peer window $[t - 30\text{d}, t + 30\text{d}]$ against a retrospective $[t - 60\text{d}, t)$ window.
10. **Stage 10 (`10_congestion_test.py`): State-Dependent Peer Congestion Test:** Evaluates the state-dependent marginal peer congestion effect $E(x) = b_{3\_u} + b_{5\_u} x$ across the observed support $x \in [0.00, 0.68]$, calculating delta-method standard errors and cluster-pairs bootstrap confidence intervals.

---

## 3. Core Structural Payoff Specifications

### Econometric Estimation Model (Centered Model 4)

$$
\ln(V_{it}) = \beta_0^* + \beta_1^* (x_{it} - \mu_x) + \beta_2^* (x_{it} - \mu_x)^2 + \beta_3^* (\bar{x}_{-i,t} - \mu_{\bar{x}}) + \beta_5^* (x_{it} - \mu_x)(\bar{x}_{-i,t} - \mu_{\bar{x}}) + \gamma_s \ln(s_i) + \gamma_d \ln(\text{dur}_{it}) + \gamma_a \text{age}_{it} + \varepsilon_{it}
$$

- Sample centering means: $\mu_x = 0.392395$, $\mu_{\bar{x}} = 0.391390$.
- Standard deviation of own appeal: $\mathrm{SD}(x) = 0.1037$.
- Standard deviation of peer appeal: $\mathrm{SD}(\bar{x}) = 0.0074$ (14x smaller, causing weak peer identification).

### Game-Theoretic Payoff Function (Uncentered Paper Eq. 2)

$$
\pi_i(x_i, \bar{x}_{-i}) = b_{0,u} + b_{1,u} x_i + b_{2,u} x_i^2 + b_{3,u} \bar{x}_{-i} + b_{5,u} x_i \bar{x}_{-i} + \gamma_s \ln(s_i) + \gamma_d \ln(\text{dur}_{it}) + \gamma_a \text{age}_{it}
$$

Analytically derived from centered coefficients with verified prediction equivalence ($\max |\hat{y}_c - \hat{y}_u| < 10^{-14}$):
- $b_{1,u} = b_{1c} - 2 b_2 \mu_x - b_5 \mu_{\bar{x}}$
- $b_{3,u} = b_{3c} - b_5 \mu_x$
- $b_{2,u} = b_2, \quad b_{5,u} = b_5$
- $b_{0,u} = b_{0c} - b_{1c}\mu_x + b_2 \mu_x^2 - b_{3c}\mu_{\bar{x}} + b_5 \mu_x \mu_{\bar{x}}$

---

## 4. Key Empirical & Theoretical Findings

### Model Progression & Out-of-Sample Performance Comparison

```
+------------------------+-------------+-------------+-------------+-------------+
| Model                  | Train R^2   | LOCO-CV R^2 | Val R^2_OOS | Test R^2_OOS |
|                        | (N=541)     | (26 folds)  | (N=136)     | (N=132)     |
+------------------------+-------------+-------------+-------------+-------------+
| Model 0 (Controls Base)|   0.6324    |   0.5741    |   0.4768    |   0.7535    |
| Model 1 (Linear Appeal)|   0.6373    |   0.5708    |   0.4035    |   0.7473    |
| Model 2 (Quadratic)    |   0.6376    |   0.5600    |   0.3836    |   0.7519    |
| Model 3 (Peer Congest) |   0.6378    |   0.5589    |   0.3881    |   0.7534    |
| Model 4 [LOCKED SPEC]  |   0.6383    |   0.5565    |   0.3965    |   0.7537    |
| Model 5 (Full Tags)    |   0.6409    |   0.5440    |   0.4210    |   0.7559    |
+------------------------+-------------+-------------+-------------+-------------+
```

> [!IMPORTANT]
> **Model Selection Disclosure:**  
> Model 0 (controls alone) outperforms Model 4 in both LOCO-CV ($0.5741$ vs. $0.5565$) and Validation ($0.4768$ vs. $0.3965$). In Test holdout, Model 4 ($0.7537$) beats Model 0 ($0.7535$) by only $+0.0002$ ($0.02$ percentage points). Model 4-Centered was locked **strictly by theoretical necessity** because it is the minimal specification containing the structural packaging terms required for game-theoretic analysis.

### Structural Parameter Estimates ($N = 809, G = 38$)

| Parameter | Centered Estimate | CRSE SE | Student-$t$ $p$ | Uncentered Est. ($b_u$) | Delta SE | Student-$t$ $p$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Intercept ($b_0$)** | $-2.7904$ | $2.4168$ | $0.2557$ | $-1.7604$ | -- | -- |
| **Own Appeal ($b_1$)** | $-0.3135$ | $1.9191$ | $0.8711$ | $-2.0190$ | $20.9201$ | $0.9236$ |
| **Curvature ($b_2$)** | $-8.1974$ | $8.2994$ | $0.3297$ | $-8.1974$ | $8.2994$ | $0.3297$ |
| **Peer Congestion ($b_3$)**| $+2.6172$ | $6.0386$ | $0.6672$ | $-5.5424$ | $22.8522$ | $0.8097$ |
| **Interaction ($b_5$)** | $+20.7942$| $60.2260$| $0.7318$ | $+20.7942$| $60.2260$| $0.7318$ |
| **Log Subscribers ($\gamma_s$)** | $+1.0417^{***}$ | $0.1147$ | $<0.0001$ | $+1.0417^{***}$ | $0.1147$ | $<0.0001$ |
| **Log Duration ($\gamma_d$)** | $+0.0454$ | $0.2534$ | $0.8588$ | $+0.0454$ | $0.2534$ | $0.8588$ |
| **Age Days ($\gamma_a$)** | $+0.0003$ | $0.0003$ | $0.3178$ | $+0.0003$ | $0.0003$ | $0.3178$ |

Significance marker: ***p < 0.001. Degrees of freedom df = G - 1 = 37.

### Hypothesis Tests & Hypothesis Support Status
- **Joint Packaging Significance:** $F(4, 37) = 0.4739$ ($p = 0.7546$). Controls explain $99.3\%$ of the variance; packaging terms are jointly insignificant.
- **Diminishing Returns ($H_0: \beta_2 \ge 0$):** Wild Cluster Bootstrap one-sided $p = 0.2600$. Retained at $\alpha = 0.05$. Concavity is not statistically confirmed.
- **Strategic Interaction ($H_0: \beta_5 = 0$):** Wild Cluster Bootstrap two-sided $p = 0.7795$. Retained at $\alpha = 0.05$.
- **Original Proposal Hypotheses (H1-H4):** Not supported by the data:
  - *H1 ($b_1 > 0$):* not supported ($p = 0.92$) (uncentered point estimate $b_{1,u} = -2.0190$, $\mathrm{SE} = 20.9201$).
  - *H2 ($b_2 < 0$):* Insignificant ($p = 0.3297$).
  - *H3 ($b_3 < 0$):* Insignificant ($p = 0.8097$).
  - *H4 ($b_5 < 0$):* Point estimate is positive ($+20.7942$, complements) and insignificant ($p = 0.7318$).

### Cluster-Pairs Bootstrap Diagnostics ($B = 2{,}000$)
- **Standard Error Verification:** Ratio $`\text{SE}_{\text{boot}} / \text{SE}_{\text{CRSE}}`$ is between $0.95$ and $1.06$ for all variables (log-subscribers ratio is $1.064$, resolving the earlier 12x deflation artifact).
- **Concave Draws ($b_2 < 0$):** $81.40\%$ (1,628 / 2,000).
- **Contraction & Concave Satisfied ($\rho < 1$ and $b_2 < 0$):** Exactly $18.65\%$ (373 / 2,000).
- **Contraction Ratio ($\rho = |b_5| / (2|b_2|)$ among concave draws):**
  - Point estimate: $\rho = 1.2683$
  - 95% Confidence Interval: $[0.1140, 35.7036]$
  - Median: $2.1528$ ($p_{10} = 0.4200, p_{25} = 1.1053, p_{75} = 4.0546, p_{90} = 9.4619$).

---

## 5. Output for Downstream Use (Game-Theoretic Parameters)

This section provides the calibrated empirical structural parameters and covariance structure for downstream game-theoretic modeling.

### Primary Structural Parameters (Paper Eq. 2)
The uncentered empirical payoff function parameters corresponding to paper Equation (2) are:

$$
\pi_i(x_i, \bar{x}_{-i}) = b_{0,u} + b_{1,u} x_i + b_{2,u} x_i^2 + b_{3,u} \bar{x}_{-i} + b_{5,u} x_i \bar{x}_{-i} + \gamma_s \ln(s_i) + \gamma_d \ln(\text{dur}_{it}) + \gamma_a \text{age}_{it}
$$

- **Uncentered Point Estimates:**  
  $b_{1,u} = -2.0190, \quad b_{2,u} = -8.1974, \quad b_{3,u} = -5.5424, \quad b_{5,u} = +20.7942$  
  $b_{0,u} = -1.7604, \quad \gamma_s = 1.0417, \quad \gamma_d = 0.0454, \quad \gamma_a = 0.0003$
- **Centering Sample Means:**  
  $\mu_x = 0.392395, \quad \mu_{\bar{x}} = 0.391390$
- **Delta-Method Covariance Matrix $\mathbf{\Sigma}_u$ (Order: $b_{1,u}, b_{2,u}, b_{3,u}, b_{5,u}$):**

$$
\mathbf{\Sigma}_u = \begin{pmatrix}
437.6519 & 81.7337 & 456.3322 & -1233.9756 \\
81.7337 & 68.8802 & 108.1554 & -316.4174 \\
456.3322 & 108.1554 & 522.2211 & -1330.6057 \\
-1233.9756 & -316.4174 & -1330.6057 & 3627.1741
\end{pmatrix}
$$

### Factual Mathematical Properties
Factually, at the empirical point estimates:
- $b_{5,u} > 2|b_{2,u}|$ ($20.7942 > 16.3948$) and $b_{2,u} + b_{5,u} > 0$ ($-8.1974 + 20.7942 = +12.5968 > 0$).
- Across $2{,}000$ cluster-pairs bootstrap draws, only $18.65\%$ of draws satisfy $|b_5| < 2|b_2|$.

---

## 6. Directory Structure & Complete Artifact Catalog

```text
payoff_calculation/
|--  data/
|   |--  processed_payoff_data.csv             # Cleaned, validated dataset (N=809)
|   |--  data_dictionary.csv                   # Complete 21-variable definitions
|   |--  channel_splits.json                   # Stratified channel partition map (seed=42)
|   \--  partitioned_payoff_data.csv           # Full dataset labeled with partition
|--  scripts/
|   |--  01_prepare_payoff_data.py             # Stage 1: Ingestion & audit
|   |--  02_create_stratified_split.py         # Stage 2: Stratified split (26/6/6)
|   |--  03_train_models.py                    # Stage 3: Training regressions & robustness
|   |--  04_evaluate_loco_cv.py                # Stage 4: 26-fold LOCO-CV on training cohort
|   |--  05_validate_and_freeze.py             # Stage 5: Validation evaluation & model locking
|   |--  06_evaluate_test_holdout.py           # Stage 6: Holdout test evaluation
|   |--  07_post_test_structural_calibration.py# Stage 7: Full estimation, uncentering & bootstrap
|   |--  08_generate_plots.py                  # Stage 8: Publication figure generation
|   |--  09_backward_looking_robustness.py     # Stage 9: Backward-looking peer window check
|   \--  10_congestion_test.py                 # Stage 10: State-dependent peer congestion test E(x)
|--  results/
|   |--  game_theory_coefficients.json         # Complete structural payload for downstream game theory
|   |--  game_theory_coefficients.csv          # Flat structural parameter summary
|   |--  post_test_structural_regression_results.csv # Full-sample regression table
|   |--  bootstrap_inference_results.csv       # Bootstrap CIs & contraction stats
|   |--  bootstrap_draws.npz                   # 2,000 cluster-pairs bootstrap parameter draws
|   |--  congestion_test.csv                   # Stage 10: Congestion function E(x) grid & bootstrap results
|   |--  training_regression_results.csv       # Training model estimates
|   |--  training_model_comparison.csv         # Training AIC, BIC, R^2
|   |--  training_multicollinearity_diagnostics.csv # Centered vs uncentered VIFs
|   |--  subscriber_robustness_table.csv       # Scale and FE robustness specifications
|   |--  loco_cv_aggregate_summary.csv         # Stage 4 LOCO-CV summary
|   |--  loco_cv_fold_results.csv              # Stage 4 fold-by-fold results
|   |--  validation_evaluation.csv             # Stage 5 validation evaluation
|   |--  validation_channel_dispersion.csv     # Stage 5 validation channel breakdown
|   |--  test_set_evaluation.csv               # Stage 6 test evaluation
|   |--  test_channel_dispersion.csv           # Stage 6 test channel breakdown
|   \--  backward_looking_peer_robustness.csv  # Contemporaneous vs backward peer table
|--  plots/
|   |--  01_model_progression_fit.png          # Figure 1: In-sample vs OOS R^2 and AIC/BIC
|   |--  02_actual_vs_predicted.png            # Figure 2: Actual vs predicted log views
|   |--  03_residual_audit.png                 # Figure 3: Four-panel residual diagnostics
|   |--  04_payoff_response_curves.png         # Figure 4: Payoff response under production costs
|   |--  05_peer_interaction_heatmap.png       # Figure 5: Strategic contour surface & BR path
|   |--  06_channel_error_dispersion.png       # Figure 6: Channel-level error dispersion
|   \--  07_congestion_test.png                # Figure 7: State-dependent peer congestion function E(x)
|--  tests/
|   |--  test_bootstrap_verification.py        # Automated test for cluster-pairs bootstrap
|   |--  test_report_consistency.py           # Automated test verifying report numerical accuracy
|   |--  test_equilibrium_consistency.py       # Structural transformation & parameter consistency
|   \--  test_encoding.py                      # UTF-8 and mojibake audit test
|--  _extras/                                  # Archived downstream equilibrium scripts, sweeps & legacy files
|--  payoff_estimation.py                     # Master execution runner
\--  README.md                                  # This document
```

---

## 7. End-to-End Reproduction Instructions

To execute all 10 empirical stages in clean sequence:

```bash
cd payoff_calculation
python payoff_estimation.py
```

To run the automated verification test suite:
```bash
python tests/test_bootstrap_verification.py
python tests/test_report_consistency.py
```
*Total execution completes in approximately 45 seconds.*
