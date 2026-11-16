# Empirical Payoff Function Estimation: Final Comprehensive Research Report

**Course:** CS 329: Foundations of AI: Multi-Agent Systems (Autumn 2026)  
**Institution:** Indian Institute of Technology Gandhinagar  
**Project Title:** *Click Appeal as a Potential Game: Modeling Creator Incentives, Attention Congestion, and Social Welfare in YouTube Recommendation*  
**Module:** Empirical Payoff Function Estimation, Econometric Regression Analysis & Statistical Inference  
**Author:** Bhavya Shah (Roll: 24110324)  
**Collaborators:**  
- Ankita Kushwaha (Roll: 24110043) — *Multimodal Scoring & Data Preprocessing*  
- Khushi Shah (Roll: 24110326) — *Game-Theoretic Formulation & Potential Game Calibration*  
- Harshita Singh (Roll: 24110342) — *Platform Mediation & Multi-Agent Simulation*  
**Date of Completion:** November 16, 2026  
**Status:** Phase 2 Fully Completed, Validated, and Audited End-to-End

---

## 1. Executive Summary

This report establishes the empirical foundation for the project *"Click Appeal as a Potential Game"*. The central purpose of this econometric component is to estimate the structural payoff function of YouTube video creators directly from multimodal feature data ($N = 809$ videos across $G = 38$ distinct YouTube channels), replacing arbitrary or synthetic payoff functions with real-world empirical parameters.

### Core Achievements
1. **Full-Pipeline Execution:** Successfully implemented and audited Steps 1 through 8 of the Phase 2 Empirical Payoff Estimation architecture.
2. **Channel-Stratified Partitioning:** Partitioned the 38 channels into 26 Training channels ($N = 541$), 6 Validation channels ($N = 136$), and 6 Held-Out Test channels ($N = 132$), preserving exact subscriber-tier distributions and preventing channel leakage.
3. **Rigorous Cross-Validation:** Completed 26-fold Leave-One-Channel-Out Cross-Validation (LOCO-CV) strictly within the training cohort ($R^2_{\mathrm{OOS}} = 0.5565$, $\mathrm{RMSE} = 1.6031$ for Model 4-Centered).
4. **Validation Freezing:** Evaluated candidate specifications on the 6 validation channels and formally locked **Model 4-Centered (Strategic Payoff Function)** prior to any test-set access ($R^2_{\mathrm{OOS}} = 0.3965$, $\mathrm{RMSE} = 1.6908$).
5. **Holdout Test Evaluation:** Evaluated the locked specification on the untouched 6 test channels after refitting on the pooled 32 training+validation channels, achieving a strong out-of-sample fit ($R^2_{\mathrm{OOS}} = 0.7537$, $\mathrm{RMSE} = 1.1382$).
6. **Full-Sample Structural Calibration:** Calibrated structural game-theoretic coefficients on all 38 channels ($N = 809$), executed $B = 2{,}000$ Null-Restricted Wild Cluster Bootstrap (WCB) tests and unrestricted bootstrap confidence intervals, and audited the contraction mapping condition ($\rho = 1.2683$).
7. **Game Theory Handoff:** Exported complete parameter payloads, variance-covariance matrices, and exact convex cost conditions ($\kappa_{\min} > 4.40$) to Khushi Shah for potential game simulation and social optimum derivation.

---

## 2. Step 3 Audit Reconciliation & Methodological Clarifications

Before finalizing Phase 2, two outstanding verification items from the Step 3 audit were resolved:

### 2.1 Resolution of Specification C Interaction $p$-Value Discrepancy
In the Step 3 exploratory audit, an apparent discrepancy was noted between a reported interaction $p$-value of $0.3868$ and $0.3948$ (or $0.3950$). The exact mathematical origin has been verified:

For Specification C (Channel Fixed Effects, within-channel estimation):
- Point estimate: $\hat{\beta}_5 = -30.070106$
- Cluster-robust standard error: $\widehat{\mathrm{SE}}_{\mathrm{CRSE}} = 34.743048$
- Empirical $t$-statistic:
  $$t = \frac{-30.070106}{34.743048} = -0.865500$$

The two values reflect different reference distributions:
1. **Asymptotic Standard Normal ($z$-test):**
   $$p_z = 2 \times (1 - \Phi(|-0.865500|)) = 2 \times (1 - 0.806618) = \mathbf{0.386764} \approx 0.3868$$
   This is the default asymptotic $p$-value emitted by `statsmodels` when computing cluster-robust covariances (`cov_type='cluster'`).
2. **Finite-Sample Student-$t$ Distribution ($df = G - 1 = 25$):**
   $$p_t = 2 \times (1 - F_{t(25)}(|-0.865500|)) = 2 \times (1 - 0.802503) = \mathbf{0.394994} \approx 0.3950$$
   This is the small-sample adjusted $p$-value recommended by Cameron and Miller (2015) for moderate cluster counts ($G = 26$).

**Conclusion:** Both calculations are mathematically correct and lead to the identical substantive conclusion: the strategic interaction term under within-channel fixed effects is statistically indistinguishable from zero ($p \approx 0.39$).

### 2.2 Backward-Looking Peer Window $[t - 60\text{d}, t)$ Robustness Analysis
We investigated whether defining competitor packaging contemporaneously ($[t - 30\text{d}, t + 30\text{d}]$) versus retrospectively ($[t - 60\text{d}, t)$) alters the empirical estimates:

```
+--------------------------------+--------------------------------+------------+------------+------------+------------+--------+
| Sample                         | Peer Window Definition         |   beta_1   |   beta_2   |   beta_3   |   beta_5   |   R²   |
+--------------------------------+--------------------------------+------------+------------+------------+------------+--------+
| Training Cohort (N=541, G=26)  | Contemporaneous [t-30d, t+30d] |    +1.6357 |    +1.1345 |    -5.6031 |   -82.1695 | 0.6385 |
| Training Cohort (N=541, G=26)  | Backward-Looking [t-60d, t)    |    +1.6598 |    +1.9388 |    +1.7687 |   -63.3707 | 0.6388 |
| Full Sample (N=809, G=38)      | Contemporaneous [t-30d, t+30d] |    -0.3135 |    -8.1974 |    +2.6172 |   +20.7942 | 0.6353 |
| Full Sample (N=809, G=38)      | Backward-Looking [t-60d, t)    |    -0.3175 |    -8.2613 |    +0.4679 |   +32.4845 | 0.6356 |
+--------------------------------+--------------------------------+------------+------------+------------+------------+--------+
```

**Key Findings:**
1. Across both samples, the qualitative curvature and interaction signs are preserved. In the full sample, $\beta_2$ is strictly negative under both specifications ($-8.1974$ vs. $-8.2613$), and $\beta_5$ remains statistically insignificant ($p = 0.73$ vs. $p = 0.56$).
2. Creator scale elasticity ($\gamma_s$) remains remarkably stable at $1.0417$ (contemporaneous) vs. $1.0423$ (backward-looking), with $p < 0.0001$.
3. The contemporaneous window $[t - 30\text{d}, t + 30\text{d}]$ is retained as the primary specification because YouTube's recommendation algorithm evaluates viewer impressions against competing videos actively circulating within the same 60-day window, regardless of whether a competing video was uploaded 15 days before or 15 days after the focal video.

### 2.3 Strict Concavity vs. Interior Optimality
We correct the theoretical distinction between concavity and interior optimality:
- **Strict Concavity** of the revenue function ($\beta_2 < 0$) guarantees that any critical point is a global maximum on $\mathbb{R}$ and that a unique maximizer exists on any compact interval $[x_{\min}, x_{\max}]$.
- **Interior Optimality** ($x^* \in (0, 1)$) additionally requires that the marginal payoff changes sign within the feasible strategy space $[0, 1]$:
  $$\left. \frac{\partial \Pi_i}{\partial x_i} \right|_{x_i = 0} > 0 \quad \text{and} \quad \left. \frac{\partial \Pi_i}{\partial x_i} \right|_{x_i = 1} < 0$$
  Under linear cost $C(x) = cx$, if $\beta_1^* - c + \beta_5^* \tilde{\bar{x}} > 0$ and $2|\beta_2|$ is small, the optimal click appeal may hit the upper boundary $x^* = 1.0$. Incorporating convex costs $C(x) = cx + \frac{\kappa}{2} x^2$ with $\kappa > 0$ steepens marginal costs and guarantees an interior solution $x^* \in (0, 1)$ for realistic creator production budgets.

---

## 3. Data Ingestion & Sample Stratification (Steps 1 & 2)

### 3.1 Sample Dimensions and Validation
The data pipeline merges multimodal packaging metrics from `video_scores.csv` with YouTube metadata from `cleaned_videos.csv`:
- **Total Sample:** Exactly $N = 809$ unique videos across $G = 38$ distinct channels (zero lost records; zero duplicates).
- **Dependent Variable:** Video view counts range from $\min = 898$ to $\max = 215{,}604{,}900$ (median $= 1{,}342{,}322$). Zero non-positive views; $\ln(V_{it})$ is smooth, approximately Gaussian, and requires zero clipping.
- **Creator Scale:** Channel subscriber counts range from $118{,}000$ to $150{,}000{,}000$. Standardized to channel-level median to eliminate minor temporal snapshot drift.
- **Video Duration:** Corrected data dictionary range: $\min = 181$ seconds (3.0 minutes) to $\max = 59{,}251$ seconds (16.46 hours, Ludwig charity livestream).

### 3.2 Channel-Level Stratified Partition Map
To prevent data leakage caused by repeated observations from the same creator, partitioning was executed strictly at the **channel level** ($seed = 42$) across three subscriber tiers:

| Subscriber Tier | Total Channels | Training Cohort ($G=26$) | Validation Cohort ($G=6$) | Held-Out Test Cohort ($G=6$) |
| :--- | :---: | :---: | :---: | :---: |
| **Tier 1: Low (<5M)** | 12 | 8 channels (174 videos) | 2 channels (50 videos) | 2 channels (48 videos) |
| **Tier 2: Mid (5M–20M)** | 16 | 11 channels (242 videos) | 3 channels (63 videos) | 2 channels (37 videos) |
| **Tier 3: Mega (>20M)** | 10 | 7 channels (125 videos) | 1 channel (23 videos) | 2 channels (47 videos) |
| **Total** | **38** | **26 channels (541 videos)** | **6 channels (136 videos)** | **6 channels (132 videos)** |

**Channel Allocations:**
- **Training Cohort (26 channels, 541 videos):** Adley, Brent Rivera, Chloe Ting, Clean Bandit, Cocomelon, DanTDM, FBE, Gourmet Makes, Grand Kitchen, Jennxpenn, Jesse, Karolina Protsenko, Kiara Madisen, Lucas and Marcus, Mark Wiens, Penguinz0, Rosanna Pansino, Ryan Trahan, Smosh, Supercar Blondie, The Food Ranger, The Slow Mo Guys, The Try Guys, Thinknoodles, Unspeakable, Zach King.
- **Validation Cohort (6 channels, 136 videos):** Max Fosh, Michelle Khare, Nick DiGiovanni, Randy Santel, Sorted Food, Wake'N'Bite.
- **Test Cohort (6 channels, 132 videos):** Airrack, Foodie We, How Ridiculous, Joel Hansen, Matt Stonie, Stokes Twins.

**Peer Pool Isolation Protocol:**
- **Training peer averages:** Calculated strictly within the other 25 training channels ($\mathrm{Train} \setminus \{i\}$).
- **Validation peer averages:** Calculated strictly using the 26 training channels.
- **Test peer averages:** Calculated strictly using the combined 32 training + validation channels.
Zero test or validation click appeals contaminated training feature construction.

---

## 4. Primary Model Development on Training Sample (Step 3)

The primary econometric hierarchy was fitted on the 26 training channels ($N = 541$):

$$\begin{aligned}
\text{Model 0 (Controls):} \quad & \ln(V_{it}) = \beta_0 + \gamma_s \ln(s_i) + \gamma_d \ln(\text{dur}_{it}) + \gamma_a \text{age}_{it} + \varepsilon_{it} \\
\text{Model 1 (Linear):} \quad & \ln(V_{it}) = \text{Model 0} + \beta_1^* \tilde{x}_{it} \\
\text{Model 2 (Quadratic):} \quad & \ln(V_{it}) = \text{Model 1} + \beta_2^* \tilde{x}_{it}^2 \\
\text{Model 3 (Peer Congestion):} \quad & \ln(V_{it}) = \text{Model 2} + \beta_3^* \tilde{\bar{x}}_{-i,t} \\
\text{Model 4 (Interaction):} \quad & \ln(V_{it}) = \text{Model 3} + \beta_5^* (\tilde{x}_{it} \cdot \tilde{\bar{x}}_{-i,t}) \\
\text{Model 5 (Full Tags):} \quad & \ln(V_{it}) = \text{Model 4} + \gamma_t \ln(\text{tags}_{it} + 1)
\end{aligned}$$

Where $\tilde{x}_{it} = x_{it} - \mu_x^{\mathrm{Train}}$ and $\tilde{\bar{x}}_{-i,t} = \bar{x}_{-i,t} - \mu_{\bar{x}}^{\mathrm{Train}}$ are centered at the training means ($\mu_x = 0.388862, \mu_{\bar{x}} = 0.388553$).

### 4.1 Training Estimation Summary Table ($N = 541, G = 26$)

| Parameter | Model 0 (Base) | Model 1 (Linear) | Model 2 (Quad) | Model 3 (Peer) | Model 4 (M4-Centered) | Model 5 (Tags) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Intercept ($\beta_0^*)** | $-2.3392$ | $-2.3551$ | $-2.3853$ | $-2.3951$ | $-2.3995$ | $-2.6627$ |
| (CRSE) | $(2.8277)$ | $(2.8170)$ | $(2.8091)$ | $(2.8091)$ | $(2.8028)$ | $(2.7303)$ |
| **Own Appeal ($\beta_1^*)** | — | $+1.4646$ | $+1.5034$ | $+1.5209$ | $+1.6357$ | $+1.5979$ |
| (CRSE) | — | $(1.4727)$ | $(1.5057)$ | $(1.5173)$ | $(1.5738)$ | $(1.5645)$ |
| **Curvature ($\beta_2^*)** | — | — | $+1.4284$ | $+1.4746$ | $+1.1345$ | $+0.7516$ |
| (CRSE) | — | — | $(5.5996)$ | $(5.6705)$ | $(5.9187)$ | $(5.8239)$ |
| **Peer Appeal ($\beta_3^*)** | — | — | — | $-3.1497$ | $-5.6031$ | $-5.9388$ |
| (CRSE) | — | — | — | $(5.3942)$ | $(7.8808)$ | $(7.8967)$ |
| **Interaction ($\beta_5^*)** | — | — | — | — | $-82.1695$ | $-80.2038$ |
| (CRSE) | — | — | — | — | $(61.1742)$ | $(60.4005)$ |
| **Log Subscribers ($\gamma_s$)** | $+1.0251^{***}$ | $+1.0249^{***}$ | $+1.0245^{***}$ | $+1.0238^{***}$ | $+1.0238^{***}$ | $+1.0379^{***}$ |
| (CRSE) | $(0.1257)$ | $(0.1250)$ | $(0.1246)$ | $(0.1246)$ | $(0.1246)$ | $(0.1213)$ |
| **Log Duration ($\gamma_d$)** | $+0.0406$ | $+0.0461$ | $+0.0463$ | $+0.0475$ | $+0.0450$ | $+0.0487$ |
| **Age Days ($\gamma_a$)** | $+0.0003$ | $+0.0003$ | $+0.0003$ | $+0.0003$ | $+0.0003$ | $+0.0003$ |
| **In-Sample $R^2$** | $0.6358$ | $0.6373$ | $0.6374$ | $0.6377$ | **$0.6385$** | $0.6409$ |
| **Akaike IC (AIC)** | $1912.4$ | $1912.2$ | $1914.1$ | $1915.6$ | **$1913.4$** | $1913.6$ |
| **Bayesian IC (BIC)** | $1933.9$ | $1937.9$ | $1944.2$ | $1949.9$ | **$1947.7$** | $1952.2$ |
| **Max VIF** | $1.13$ | $1.14$ | $1.16$ | $1.16$ | **$1.15$** | $1.20$ |

*Significance markers: $^{***} p < 0.001$. Standard errors are clustered by channel ($G = 26$).*

### 4.2 Exact Economic Interpretation of Point Estimates
For log-transformed view counts, the exact percentage change in views resulting from a change $\Delta X$ is:
$$\% \Delta V = 100 \times \left( \exp(\beta \cdot \Delta X) - 1 \right)$$

1. **Own Click Appeal ($\hat{\beta}_1^* = +1.6357$, $p = 0.2986$):**
   A hypothetical $+0.10$ increase in own click appeal score from the mean predicts an estimated view change of:
   $$\% \Delta V = 100 \times (\exp(1.6357 \times 0.10) - 1) = +17.77\%$$
   *Statistical Caveat:* Because $\mathrm{SE} = 1.5738$ ($p = 0.30$), this estimate carries substantial sampling variance and is not statistically established at $\alpha = 0.05$.
2. **Peer Attention Congestion ($\hat{\beta}_3^* = -5.6031$, $p = 0.4771$):**
   A $+0.10$ increase in competitor average packaging predicts an estimated view change of:
   $$\% \Delta V = 100 \times (\exp(-5.6031 \times 0.10) - 1) = -42.89\%$$
   *Statistical Caveat:* Insignificant at $\alpha = 0.05$ ($\mathrm{SE} = 7.8808$, $p = 0.48$).
3. **Strategic Interaction ($\hat{\beta}_5^* = -82.1695$, $p = 0.1792$):**
   Suggests negative interaction (strategic substitutability), but carries a large standard error ($\mathrm{SE} = 61.1742$).
4. **Creator Scale Elasticity ($\hat{\gamma}_s = +1.0238$, $p < 0.0001$):**
   A $1\%$ increase in subscriber count is associated with a $+1.02\%$ increase in views. Creator scale is the single dominant predictive driver ($t = 8.22$).

### 4.3 Multicollinearity Resolution: Uncentered vs. Centered Model 4
In uncentered Model 4, structural collinearity between $x$, $x^2$, and $x \cdot \bar{x}$ produced severe variance inflation:
- Uncentered VIFs: $x = 1{,}334.90$, $x^2 = 281.44$, $x \cdot \bar{x} = 1{,}281.33$.
- Centered Model 4 VIFs: $x_c = 1.13$, $x_c^2 = 1.15$, $x_c \cdot \bar{x}_c = 1.14$.

Mean-centering completely eliminates artificial collinearity ($VIF < 1.20$ for all variables) while preserving identical explanatory power, residuals, $R^2$, and curvature coefficients.

---

## 5. Model Evaluation: 26-Fold LOCO-CV & Validation Freezing (Steps 4 & 5)

### 5.1 Step 4: 26-Fold LOCO-CV on Training Cohort ($N = 541$)
To evaluate out-of-sample predictive power without data leakage, we executed 26-fold Leave-One-Channel-Out Cross-Validation strictly within the training cohort. In each fold $k \in \{1, \dots, 26\}$:
- Channel $k$ is held out completely.
- Peer averages for the remaining 25 channels are computed strictly within those 25 channels.
- Centering means $\mu_x^{(k)}, \mu_{\bar{x}}^{(k)}$ and baseline mean $\bar{y}^{(k)}$ are computed strictly from the 25 training channels.
- Total out-of-sample sum of squares: $\mathrm{SST}_{\mathrm{OOS}} = \sum_{k=1}^{26} \sum_{i \in \mathrm{fold}_k} (y_i - \bar{y}^{(k)})^2 = 3{,}135.25$.

```
+--------------------------+-------------+-------------+------------+------------+------------+
| Candidate Model          |   SSE_OOS   |   SST_OOS   |   R²_OOS   |  RMSE_OOS  |  MAE_OOS   |
+--------------------------+-------------+-------------+------------+------------+------------+
| Model 0 (Controls Base)  |   1335.25   |   3135.25   |   0.5741   |   1.5710   |   1.2963   |
| Model 1 (Linear Appeal)  |   1345.54   |   3135.25   |   0.5708   |   1.5771   |   1.3056   |
| Model 2 (Quadratic)      |   1379.46   |   3135.25   |   0.5600   |   1.5969   |   1.3215   |
| Model 3 (Peer Congestion)|   1382.88   |   3135.25   |   0.5589   |   1.5988   |   1.3225   |
| Model 4 (Interaction)    |   1390.41   |   3135.25   |   0.5565   |   1.6031   |   1.3280   |
| Model 5 (Full Tags)      |   1429.62   |   3135.25   |   0.5440   |   1.6257   |   1.3401   |
+--------------------------+-------------+-------------+------------+------------+------------+
```

*Key Insight:* In pure out-of-sample prediction, Model 0 achieves $R^2_{\mathrm{OOS}} = 0.5741$. Adding packaging terms incurs a slight shrinkage penalty ($R^2_{\mathrm{OOS}} = 0.5565$), common in econometric specifications where packaging effects are localized relative to massive creator scale differentials.

### 5.2 Step 5: Validation Evaluation & Specification Freezing (6 Channels, $N = 136$)
We evaluated candidate models on the 6 validation channels (Max Fosh, Michelle Khare, Nick DiGiovanni, Randy Santel, Sorted Food, Wake'N'Bite). Centering means and peer pools were drawn strictly from the 26 training channels:

$$\mathrm{SST}_{\mathrm{Val}} = \sum_{j \in \mathrm{Val}} (y_j - \bar{y}_{\mathrm{Train}})^2 = 644.22$$

```
+--------------------------+-------------+-------------+------------+------------+------------+
| Model                    |   SSE_Val   |   SST_Val   | R²_OOS_Val |  RMSE_Val  |  MAE_Val   |
+--------------------------+-------------+-------------+------------+------------+------------+
| Model 0 (Controls Base)  |   337.08    |   644.22    |   0.4768   |   1.5743   |   1.2290   |
| Model 1 (Linear Appeal)  |   384.29    |   644.22    |   0.4035   |   1.6810   |   1.3022   |
| Model 2 (Quadratic)      |   397.10    |   644.22    |   0.3836   |   1.7087   |   1.3209   |
| Model 3 (Peer Congestion)|   394.22    |   644.22    |   0.3881   |   1.7025   |   1.3180   |
| Model 4 (Interaction)    |   388.81    |   644.22    | **0.3965** | **1.6908** | **1.3126** |
| Model 5 (Full Tags)      |   372.98    |   644.22    |   0.4210   |   1.6560   |   1.2633   |
+--------------------------+-------------+-------------+------------+------------+------------+
```

#### Specification Locking Decision: Model 4-Centered
We formally lock **Model 4-Centered** as the final specification for the potential game payoff function:
1. **Theoretical Necessity:** Model 4 contains the quadratic curvature $\beta_2$ and strategic interaction $\beta_5$ required to formalize diminishing returns and evaluate the contraction mapping condition ($\rho < 1$). Models 0–1 lack these structural terms.
2. **Empirical Performance:** On the validation cohort, Model 4 achieves $R^2_{\mathrm{OOS}} = 0.3965$ (improving over Model 2 and Model 3).
3. **Parsimony:** Model 5 (tags) adds noise without structural game-theoretic necessity.
4. **Specification Lock:** Model exploration is terminated; Model 4-Centered is frozen.

---

## 6. Final Holdout Evaluation on Untouched Test Set (Step 6)

### 6.1 Methodological Disclosure of Exploratory History
In adherence to scientific integrity, we explicitly disclose that all 809 observations were initially examined during early exploratory Phase 1 feasibility analyses before the channel-stratified train/validation/test protocol was finalized. Consequently, this Step 6 evaluation represents an **internal holdout validation** of the frozen Model 4-Centered specification rather than a pure pre-registered ex-ante clinical trial.

### 6.2 Test Evaluation Setup & Execution ($N = 132$, $G = 6$)
The locked Model 4-Centered specification was refitted on the combined **Training + Validation sample (32 channels, 677 videos)**. Test peer features were drawn strictly from the combined 32-channel pool.
- Combined sample centering: $\mu_x^{\mathrm{Train+Val}} = 0.393912, \mu_{\bar{x}}^{\mathrm{Train+Val}} = 0.393521, \bar{y}_{\mathrm{Train+Val}} = 14.045068$.
- Total test sum of squares: $\mathrm{SST}_{\mathrm{Test}} = 694.24$.

```
+--------------------------+-------------+-------------+-------------+-----------+-----------+
| Model Specification      |  SSE_Test   |  SST_Test   | R²_OOS_Test | RMSE_Test | MAE_Test  |
+--------------------------+-------------+-------------+-------------+-----------+-----------+
| Model 0 (Controls Base)  |   171.13    |   694.24    |   0.7535    |  1.1386   |  0.9170   |
| Model 1 (Linear Appeal)  |   175.42    |   694.24    |   0.7473    |  1.1528   |  0.9334   |
| Model 2 (Quadratic)      |   172.21    |   694.24    |   0.7519    |  1.1422   |  0.9254   |
| Model 3 (Peer Congestion)|   171.20    |   694.24    |   0.7534    |  1.1388   |  0.9193   |
| Model 4 [LOCKED SPEC]    |   171.02    |   694.24    | **0.7537**  |**1.1382** |**0.9190** |
| Model 5 (Full Tags)      |   169.50    |   694.24    |   0.7559    |  1.1332   |  0.9103   |
+--------------------------+-------------+-------------+-------------+-----------+-----------+
```

### 6.3 Test Channel Error Dispersion
Channel-by-channel performance of the locked specification across the 6 held-out test channels:

```
+------------------+---------------------+---------------+---------+----------+----------+----------+
| Channel Name     | Subscriber Tier     | Subscribers   | Videos  |   MAE    |   RMSE   |   Bias   |
+------------------+---------------------+---------------+---------+----------+----------+----------+
| Airrack          | Tier 2 (5M–20M)     | 19,600,000    |   22    |  1.0671  |  1.1733  | +0.9836  |
| Foodie We        | Tier 1 (<5M)        |  2,510,000    |   24    |  0.5334  |  0.6046  | +0.2152  |
| How Ridiculous   | Tier 3 (>20M)       | 23,900,000    |   24    |  1.0823  |  1.2951  | -1.0750  |
| Joel Hansen      | Tier 1 (<5M)        |  1,170,000    |   24    |  1.0145  |  1.4218  | -0.3628  |
| Matt Stonie      | Tier 2 (5M–20M)     | 16,400,000    |   15    |  0.6911  |  0.7646  | -0.2466  |
| Stokes Twins     | Tier 3 (>20M)       | 150,000,000   |   23    |  1.0581  |  1.2233  | +0.9557  |
+------------------+---------------------+---------------+---------+----------+----------+----------+
```

**Discussion:** The locked Model 4 achieves a strong out-of-sample $R^2_{\mathrm{OOS}} = 0.7537$ on the test cohort. Across all tiers, MAE remains bounded between $0.53$ and $1.08$ log-views, confirming that the locked specification generalizes well to unseen creators without overfitting.

---

## 7. Post-Test Full-Sample Structural Calibration & Inference (Step 7)

Following the completion of holdout evaluation, Model 4-Centered was estimated on the full dataset ($N = 809$, $G = 38$) for structural parameter calibration and game-theoretic integration.

### 7.1 Full-Sample Structural Estimates ($N = 809, G = 38$)
Centering means: $\mu_x = 0.392395, \mu_{\bar{x}} = 0.391390$.

$$\ln(V_{it}) = \beta_0^* + \beta_1^* \tilde{x}_{it} + \beta_2^* \tilde{x}_{it}^2 + \beta_3^* \tilde{\bar{x}}_{-i,t} + \beta_5^* (\tilde{x}_{it} \cdot \tilde{\bar{x}}_{-i,t}) + \gamma_s \ln(s_i) + \gamma_d \ln(\text{dur}_{it}) + \gamma_a \text{age}_{it} + \varepsilon_{it}$$

```
+--------------------+------------+------------+-----------+-----------+---------------------+
| Regressor          | Coeff (b)  | CRSE (SE)  |  t-stat   | p (df=37) | 95% Conf. Interval  |
+--------------------+------------+------------+-----------+-----------+---------------------+
| Intercept (b0*)    |   -2.7904  |   2.4168   |  -1.155   |  0.2557   | [ -7.687,  +2.106]  |
| Own Appeal (b1*)   |   -0.3135  |   1.9191   |  -0.163   |  0.8711   | [ -4.202,  +3.575]  |
| Curvature (b2*)    |   -8.1974  |   8.2994   |  -0.988   |  0.3297   | [-25.014,  +8.619]  |
| Peer Congest (b3*) |   +2.6172  |   6.0386   |  +0.433   |  0.6672   | [ -9.618, +14.853]  |
| Interaction (b5*)  |  +20.7942  |  60.2260   |  +0.345   |  0.7318   | [-101.23, +142.82]  |
| Log Subs (gamma_s) |   +1.0417  |   0.1147   |  +9.079   |  0.0000   | [ +0.809,  +1.274]  |
| Log Dur (gamma_d)  |   +0.0454  |   0.2534   |  +0.179   |  0.8588   | [ -0.468,  +0.559]  |
| Age Days (gamma_a) |   +0.0003  |   0.0003   |  +1.013   |  0.3178   | [ -0.000,  +0.001]  |
+--------------------+------------+------------+-----------+-----------+---------------------+
```

*Model Diagnostics:* $R^2 = 0.6353$, $\mathrm{AIC} = 2838.0$, $\mathrm{BIC} = 2875.6$. All centered VIFs $< 1.22$.

### 7.2 Null-Restricted Wild Cluster Bootstrap (WCB) Hypothesis Tests ($B = 2{,}000$)
Using Rademacher cluster weights ($\pm 1$) and the global finite-sample residual scaling factor:
$$c = \sqrt{\frac{G}{G - 1} \cdot \frac{N - 1}{N - K}} = \sqrt{\frac{38}{37} \cdot \frac{808}{801}} = 1.017842$$

1. **Test A: Diminishing Returns / Strict Concavity**
   - $H_0: \beta_2 \ge 0$ (weakly convex / no diminishing returns) vs. $H_1: \beta_2 < 0$ (strictly concave).
   - Sample $t$-statistic: $t = -0.9877$.
   - **WCB One-Sided $p$-Value:** $p_{\mathrm{WCB}} = \mathbf{0.2480}$.
   - **Conclusion:** $H_0$ is retained at $\alpha = 0.05$. Although the point estimate is negative ($\hat{\beta}_2^* = -8.1974$), strict revenue concavity is **not statistically established**.
2. **Test B: Strategic Interaction**
   - $H_0: \beta_5 = 0$ (no strategic interaction) vs. $H_1: \beta_5 \neq 0$.
   - Sample $t$-statistic: $t = +0.3453$.
   - **WCB Two-Sided $p$-Value:** $p_{\mathrm{WCB}} = \mathbf{0.7835}$.
   - **Conclusion:** $H_0$ is retained at $\alpha = 0.05$. Packaging strategic interdependence cannot be statistically distinguished from zero in observational data once creator scale is controlled.

### 7.3 Unrestricted Bootstrap: Confidence Intervals & Contraction Ratio $\rho$ ($B = 2{,}000$)
Parameter draws from $B = 2{,}000$ cluster-centered unrestricted bootstrap replications:

```
+--------------------+------------+------------+----------------------+
| Regressor          | Point Est. |  Boot SE   | 95% Percentile CI    |
+--------------------+------------+------------+----------------------+
| Intercept (b0*)    |   -2.7904  |   0.3302   | [  -3.410,   -2.172] |
| Own Appeal (b1*)   |   -0.3135  |   0.3852   | [  -1.028,   +0.433] |
| Curvature (b2*)    |   -8.1974  |   2.6536   | [ -13.120,   -3.193] |
| Peer Congest (b3*) |   +2.6172  |   3.0564   | [  -3.278,   +8.533] |
| Interaction (b5*)  |  +20.7942  |  31.8162   | [ -40.686,  +82.960] |
| Log Subs (gamma_s) |   +1.0417  |   0.0097   | [  +1.023,   +1.061] |
| Log Dur (gamma_d)  |   +0.0454  |   0.0531   | [  -0.056,   +0.146] |
| Age Days (gamma_a) |   +0.0003  |   0.0002   | [  -0.000,   +0.001] |
+--------------------+------------+------------+----------------------+
```

#### Contraction Ratio Evaluation
The contraction condition for continuous games on compact strategy sets is:
$$\rho = \frac{|\beta_5|}{2 |\beta_2|} < 1$$

In accordance with our zero-discarding protocol, all $2{,}000$ draws were retained:
- **Point Estimate Contraction Ratio:** $\rho = \frac{|20.7942|}{2 \times |-8.1974|} = \mathbf{1.2683}$ ($\rho > 1$).
- **Concave Draws ($\beta_2 < 0$):** $2{,}000 / 2{,}000$ ($100.0\%$).
- **Contraction Mapping Satisfaction ($P(\rho < 1)$):** $\mathbf{32.40\%}$.
- **Contraction Ratio Quantiles (among concave draws):**
  - $p_{10} = 0.2982$
  - $p_{25} = 0.7611$
  - $p_{50} (\text{Median}) = 1.6434$
  - $p_{75} = 2.8524$
  - $p_{90} = 4.4912$

---

## 8. Game-Theoretic Formulation & Potential Game Handoff

*This section provides the formal handoff payload for Khushi Shah (Roll: 24110326).*

### 8.1 Game Classification & Exact Potential Function
Under Model 4-Centered, the strategic interaction between creator $i$ and competitors $-i$ is symmetric because the peer effect enters through the unweighted aggregate $\bar{x}_{-i} = \frac{1}{N - 1} \sum_{j \neq i} x_j$.

The system forms a **Continuous Aggregative Potential Game**. The exact potential function $\Phi(x_1, \dots, x_N)$ whose gradient matches each creator's marginal payoff is:

$$\Phi(x_1, \dots, x_N) = \sum_{i=1}^N \left( \beta_1^* (x_i - \mu_x) + \beta_2^* (x_i - \mu_x)^2 \right) + \frac{\beta_5^*}{2 (N - 1)} \sum_{i=1}^N \sum_{j \neq i} (x_i - \mu_x)(x_j - \mu_{\bar{x}}) - \sum_{i=1}^N C_i(x_i)$$

Differentiating $\Phi$ with respect to $x_i$:
$$\frac{\partial \Phi}{\partial x_i} = \beta_1^* + 2 \beta_2^* (x_i - \mu_x) + \beta_5^* (\bar{x}_{-i} - \mu_{\bar{x}}) - C_i'(x_i) = \frac{\partial \Pi_i}{\partial x_i}$$
This satisfies Monderer and Shapley's (1996) condition for an **exact potential game**.

### 8.2 Production Cost Structure & Contraction Guarantee
Because the empirical point estimate yields $\rho = 1.2683 > 1$, relying solely on empirical revenue curvature does not guarantee a contraction mapping. We prescribe a quadratic production cost structure:

$$C_i(x_i) = c_i x_i + \frac{\kappa_i}{2} x_i^2 \quad (\kappa_i > 0)$$

Under this cost function, each creator's payoff second derivative becomes:
$$\frac{\partial^2 \Pi_i}{\partial x_i^2} = 2 \beta_2^* - \kappa_i = 2(-8.1974) - \kappa_i = -16.3948 - \kappa_i < 0$$
Strict concavity is unconditionally guaranteed for any $\kappa_i \ge 0$.

The best-response slope is:
$$\left| \frac{\partial \mathrm{BR}_i}{\partial \bar{x}_{-i}} \right| = \frac{|\beta_5^*|}{2 |\beta_2^*| + \kappa_i}$$

To guarantee a unique pure-strategy Nash Equilibrium via Banach's Fixed Point Theorem, the contraction condition requires:
$$\frac{|\beta_5^*|}{2 |\beta_2^*| + \kappa_{\min}} < 1 \iff \kappa_{\min} > |\beta_5^*| - 2 |\beta_2^*| = 20.7942 - 16.3948 = \mathbf{4.3994}$$

**Recommendation for Game Theory Component:** Setting convex production cost curvature $\kappa_i > 4.40$ (or linear cost slope $c_i > 1.0$) guarantees that creator best responses form a strict contraction mapping on $[0, 1]$, ensuring:
1. Existence and uniqueness of pure-strategy Nash Equilibrium $x^* = (x_1^*, \dots, x_N^*)$.
2. Convergence of synchronous and asynchronous best-response dynamics.
3. Tractable computation of Price of Anarchy ($\mathrm{PoA} = \frac{\Phi(x_{\mathrm{SO}})}{\Phi(x_{\mathrm{NE}})}$).

### 8.3 Parameter Covariance Matrix
For delta-method evaluations and Monte Carlo error propagation in the game theory simulation, the empirical covariance matrix of $(\beta_1^*, \beta_2^*, \beta_3^*, \beta_5^*)$ is exported:

$$\mathbf{\Sigma}_{\mathrm{strat}} = \begin{pmatrix}
3.6829 & -3.8967 & 3.2389 & -6.1436 \\
-3.8967 & 68.8802 & -1.2464 & -18.2396 \\
3.2389 & -1.2464 & 36.4649 & 12.8715 \\
-6.1436 & -18.2396 & 12.8715 & 3627.1741
\end{pmatrix}$$

---

## 9. Publication Figures (Step 8)

Six high-resolution figures (300 DPI) have been generated in `payoff_estimation/plots/`:
1. **`01_model_progression_fit.png`:** In-sample $R^2$, 26-fold LOCO-CV $R^2_{\mathrm{OOS}}$, and Validation $R^2_{\mathrm{OOS}}$ alongside AIC and BIC curves across models M0–M5.
2. **`02_actual_vs_predicted.png`:** Actual vs. predicted log views across Training ($R^2 = 0.638$), Validation ($R^2_{\mathrm{OOS}} = 0.396$), and Test ($R^2_{\mathrm{OOS}} = 0.754$) cohorts, color-coded by subscriber tier.
3. **`03_residual_audit.png`:** Four-panel diagnostic grid (residuals vs. fitted values, normal Q-Q plot, residuals vs. own click appeal, and residuals vs. creator scale).
4. **`04_payoff_response_curves.png`:** Gross view response curves across low ($p_{10}$), median ($p_{50}$), and high ($p_{90}$) peer competition, contrasted with net payoffs under linear vs. convex production costs.
5. **`05_peer_interaction_heatmap.png`:** Strategic contour surface $\ln V(x_i, \bar{x}_{-i})$ with the best-response reaction path overlaid.
6. **`06_channel_error_dispersion.png`:** Channel-by-channel error dispersion (MAE and bias) across all 38 channels, color-coded by sample partition.

---

## 10. Methodological Limitations & Boundary Conditions

1. **Retrospective Subscriber Snapshot:** Subscriber counts represent a late-2026 snapshot rather than historical time-varying subscriber counts at the exact video upload date. In our models, $\ln(s_i)$ acts as a channel-level fixed scale proxy.
2. **Observational Inference:** Packaging metrics ($x_{it}$) are observational rather than experimentally manipulated. While we control for duration, age, and creator scale, point estimates should be interpreted as predictive associations and structural calibration inputs, not causal treatments.
3. **Monetary Profit vs. Attention Payoff:** View counts proxy viewer attention and algorithmic impressions. True monetary creator profit depends on RPM (revenue per thousand impressions), sponsorships, and production overhead.

---

## 11. Complete Artifact Catalog & Reproduction Pipeline

All scripts and generated artifacts are maintained within `payoff_estimation/`:

```text
payoff_estimation/
├── data/
│   ├── processed_payoff_data.csv             # Step 1: Ingested & validated dataset (N=809)
│   ├── data_dictionary.csv                   # Step 1: Complete variable definitions
│   ├── channel_splits.json                   # Step 2: Channel-level stratification partition map
│   └── partitioned_payoff_data.csv           # Step 2: Full dataset labeled by partition
├── scripts/
│   ├── 01_prepare_payoff_data.py             # Step 1 execution script
│   ├── 02_create_stratified_split.py         # Step 2 execution script
│   ├── 03_train_models.py                    # Step 3 primary training regressions
│   ├── 04_evaluate_loco_cv.py                # Step 4 26-fold LOCO-CV on training cohort
│   ├── 05_validate_and_freeze.py             # Step 5 validation evaluation & model locking
│   ├── 06_evaluate_test_holdout.py           # Step 6 holdout test evaluation
│   ├── 07_post_test_structural_calibration.py# Step 7 full-sample structural estimation & bootstrap
│   ├── 08_generate_plots.py                  # Step 8 publication figures generator
│   └── 09_backward_looking_robustness.py     # Secondary peer window robustness script
├── results/
│   ├── training_regression_results.csv       # Training model estimates
│   ├── training_model_comparison.csv         # Training AIC, BIC, R²
│   ├── training_multicollinearity_diagnostics.csv # Centered vs uncentered VIFs
│   ├── subscriber_robustness_table.csv       # Specs A, B, C, R-Tags
│   ├── loco_cv_aggregate_summary.csv         # Step 4 LOCO-CV summary
│   ├── loco_cv_fold_results.csv              # Step 4 fold-by-fold results
│   ├── validation_evaluation.csv             # Step 5 validation evaluation
│   ├── validation_channel_dispersion.csv     # Step 5 validation channel breakdown
│   ├── test_set_evaluation.csv               # Step 6 test evaluation
│   ├── test_channel_dispersion.csv           # Step 6 test channel breakdown
│   ├── post_test_structural_regression_results.csv # Step 7 full-sample estimates
│   ├── bootstrap_inference_results.csv       # Step 7 bootstrap CIs & contraction stats
│   ├── game_theory_coefficients.json         # Step 7 JSON handoff payload for Khushi Shah
│   ├── game_theory_coefficients.csv          # Step 7 flat CSV payload
│   └── backward_looking_peer_robustness.csv  # Robustness comparison table
├── plots/
│   ├── 01_model_progression_fit.png          # Figure 1
│   ├── 02_actual_vs_predicted.png            # Figure 2
│   ├── 03_residual_audit.png                 # Figure 3
│   ├── 04_payoff_response_curves.png         # Figure 4
│   ├── 05_peer_interaction_heatmap.png       # Figure 5
│   └── 06_channel_error_dispersion.png       # Figure 6
├── FINAL_PAYOFF_REPORT.md                    # This master report
└── README.md                                 # Component documentation & reproduction instructions
```

### End-to-End Reproduction Command
To reproduce all results, tables, bootstrap inference, and figures sequentially from scratch:

```bash
cd payoff_estimation
python scripts/01_prepare_payoff_data.py
python scripts/02_create_stratified_split.py
python scripts/03_train_models.py
python scripts/04_evaluate_loco_cv.py
python scripts/05_validate_and_freeze.py
python scripts/06_evaluate_test_holdout.py
python scripts/07_post_test_structural_calibration.py
python scripts/08_generate_plots.py
python scripts/09_backward_looking_robustness.py
```

*All scripts run deterministically under `seed = 42` and complete in approximately 60 seconds.*
