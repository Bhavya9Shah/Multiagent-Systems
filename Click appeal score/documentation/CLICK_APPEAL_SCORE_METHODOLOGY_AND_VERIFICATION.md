# Click Appeal Score — Research Methodology and Empirical Verification Guide

**Project Title:** Click Appeal Score: Multimodal Algorithmic Quantification of YouTube Packaging  
**Author:** Undergraduate Research Assistant  
**Date:** November 2026  
**Pipeline Status:** Fully Executed, Verified, and Reconstructible (186/186 Records)  
**Execution Runtime:** Python 3.12 (`.venv` on Windows)

---

## 1. Executive Objective and Dataset Description

### 1.1 Research Objective
The **Click Appeal Score** ($X_v$) is a researcher-defined descriptive composite proxy based on selected title and thumbnail characteristics. It is not an objective or validated measure of psychological response, click-through rate (CTR), or actual click probability. The selected observable inputs include title typography and lexicon matches, face presence, FER model predictions, luminance contrast, and saturation variability.

The goal of this research project is to:
1. Build an end-to-end, reproducible computational pipeline that extracts linguistically and visually grounded features from raw YouTube metadata.
2. Formulate mathematically standardized title and thumbnail appeal indices.
3. Establish creator-level baselines using leave-one-creator-out (LOCO) benchmarks.
4. Provide a transparent, beginner-friendly mathematical audit trail that can be manually verified without black-box opacity.

### 1.2 The Dataset
The study analyzes the complete dataset provided in `challenge_videos_final(challenge_videos_final).csv`:
- **Total Records:** 186 videos.
- **Unique Creators:** 38 distinct YouTube channels (37 creators with at least 5 videos and 1 creator with 1 video).
- **Time Horizon:** Published between October 2, 2024, and October 1, 2026.
- **Domain:** High-performing challenge, competitive eating, stunt, science, and entertainment videos.
- **Variables:** `channel_name`, `channel_id`, `video_id`, `title`, `thumbnail` (URL), `published_at`, `views`, `likes`, `comments`, `duration_sec`, `age_days`, `tags_count`, `category_id`, `subscribers`, `channel_video_count`, and `face_count` (prior label).

---

## 2. Dataset Cleaning Procedure (`01_clean_data.py`)

No records were discarded or silently deleted. The dataset underwent systematic audit and cleaning:

1. **Character Encoding Standardization:**
   - Raw CSV files generated on Windows systems often contain single quotes (`’`, byte `0x92`) or pound currency symbols (`£`, byte `0xa3`) in Windows-1252 (`cp1252`) encoding.
   - The cleaning pipeline safely decodes `cp1252` and exports clean UTF-8 (`data/processed/cleaned_videos.csv`).
2. **Channel Name Inconsistency Resolution:**
   - Channel ID `UCyps-v4WNjWDnYRKmZ4BUGw` had four rows labeled `Airrack` and one row labeled `\`.
   - By matching channel ID `UCyps-v4WNjWDnYRKmZ4BUGw`, the corrupt backslash was repaired to `Airrack`.
3. **Recovery of Malformed Video IDs:**
   - In spreadsheet software (e.g., Microsoft Excel), leading hyphens `-` are frequently interpreted as formula operators, converting valid 11-character YouTube video IDs into `#NAME?`.
   - Four video IDs were corrupted in the raw CSV:
     - Row 18 (Beta Squad): `#NAME?` with thumbnail URL `https://i.ytimg.com/vi/-Ad7lJ74XSE/maxresdefault.jpg` $\rightarrow$ Recovered: `-Ad7lJ74XSE`
     - Row 47 (Guga Foods): `#NAME?` with thumbnail URL `https://i.ytimg.com/vi/-Lt0idIsJWo/maxresdefault.jpg` $\rightarrow$ Recovered: `-Lt0idIsJWo`
     - Row 83 (Leah Shutkever): `#NAME?` with thumbnail URL `https://i.ytimg.com/vi/-gnzjVbYkgE/maxresdefault.jpg` $\rightarrow$ Recovered: `-gnzjVbYkgE`
     - Row 152 (Stokes Twins): `#NAME?` with thumbnail URL `https://i.ytimg.com/vi/-j8WOAHw5d4/maxresdefault.jpg` $\rightarrow$ Recovered: `-j8WOAHw5d4`
   - All recovered IDs follow canonical 11-character YouTube alphanumeric patterns (`^[A-Za-z0-9_-]{11}$`).
4. **Audit Logs:**
   - All modifications are documented in `data/processed/cleaning_log.csv`.
   - 21 high-level quality metrics are stored in `data/processed/data_quality_report.csv`.

---

## 3. Thumbnail Retrieval Pipeline (`02_download_thumbnails.py`)

1. **Deterministic Retrieval:**
   - Thumbnails were downloaded directly from the canonical URLs specified in the CSV into `thumbnails_original/<video_id>.jpg`.
   - No mock, placeholder, or synthetic images were generated.
2. **Verification and Manifest:**
   - Every downloaded file was validated using Pillow (`Image.verify()`).
   - Dimensions, file format, byte size, and HTTP response codes were logged in `thumbnails_original/download_manifest.csv`.
   - **Results:** 186/186 successful downloads (100.0% success rate, 0 failures). 181 thumbnails are $1280 \times 720$ (HD), and 5 are $480 \times 360$ (SD/HQ).

---

## 4. Multimodal Feature Engineering

The pipeline extracts eight primary features: four linguistic title features and four computer vision thumbnail features.

```
Click Appeal Feature Space
│
├── Title Features (Linguistic)
│   ├── f_caps       : Capitalization ratio (uppercase tokens / L)
│   ├── f_trigger    : Superlative & curiosity trigger word ratio
│   ├── f_sensitive  : Taboo, risk & peril word ratio
│   └── f_symbol     : Exclamation, question mark & emoji density
│
└── Thumbnail Features (Visual)
    ├── f_face       : Binary indicator of human face presence
    ├── f_expr       : Maximum emotional confidence (Surprise vs. Fear)
    ├── f_contrast   : Population standard deviation of luminance sigma(Y)
    └── f_saturation : Population standard deviation of HSV saturation sigma(S)
```

### 4.1 Title Features

Let $L = \max(1, \text{number of whitespace-delimited tokens in title})$.

#### 1. Capitalization Ratio ($f_{caps}$)
$$\large f_{caps} = \frac{\text{Count of uppercase alphabetic tokens with} \ge 2 \text{ letters}}{L}$$
- Tokens are stripped of surrounding punctuation.
- Tokens must contain at least two letters and have all letters uppercase (e.g., `FAKE`, `UNBEATEN`, `CASH`, `AI`).
- Single-letter words (`I`, `A`) and numbers (`10,000`) do not count.

#### 2. Trigger-Word Ratio ($f_{trigger}$)
$$\large f_{trigger} = \frac{\text{Trigger word/phrase occurrences}}{L}$$
- Quantifies sensationalism, curiosity, high stakes, and superlatives.
- Utilizes documented English and Hindi baseline lexicons (see `documentation/TITLE_LEXICONS.md`).
- Multi-word phrases (e.g., `last to`, `first to`, `worth it`) are matched without double-counting.

#### 3. Sensitive-Word Ratio ($f_{sensitive}$)
$$\large f_{sensitive} = \frac{\text{Sensitive word/phrase occurrences}}{L}$$
- Measures terms denoting physical risk, controversy, crime, taboo topics, illness, and distress (e.g., `illegal`, `scam`, `heart attack`, `poison`, `dead`).

#### 4. Symbol and Emoji Ratio ($f_{symbol}$)
$$\large f_{symbol} = \frac{\text{count}(!) + \text{count}(?) + \text{emoji count}}{L}$$
- Directly counts punctuation markers of high arousal (`!`, `?`) and Unicode emojis detected via Python's `emoji` package.

---

### 4.2 Thumbnail Computer Vision Features

#### 1. Face Presence ($f_{face}$)
Face detection is executed using MediaPipe 0.10.21 (`mp.solutions.face_detection`):
- `model_selection = 1` (full-range model optimized for varied camera distances).
- `min_detection_confidence = 0.5`.
$$\large f_{face} = \begin{cases} 1 & \text{if detected face count} \ge 1 \\ 0 & \text{if detected face count} = 0 \end{cases}$$
- MediaPipe detected 296 total faces across 186 thumbnails (12 images had 0 faces, 98 had 1 face, 76 had $\ge 2$ faces).
- Bounding box coordinates and confidence scores are preserved in `features/thumbnail_features.csv`.

#### 2. Expression Arousal Score ($f_{expr}$)
Facial expressions are classified using FER 25.10.3 with a TensorFlow 2.16.1 backend:
- The MediaPipe bounding boxes are converted to pixel coordinates: $(x, y, w, h)$.
- The **largest valid face** is selected by pixel area:
  $$A_i = w_i \times h_i, \quad i^* = \arg\max_i A_i$$
- FER evaluates the probabilities of seven basic emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.
- The expression feature is defined as the following FER model output:
  $$\large f_{expr} = \max\left(P(\text{surprise}), P(\text{fear})\right)$$
- **Boundary Handling:**
  - If no face is detected ($f_{face} = 0$), $f_{expr} = 0.0$.
  - If a face is detected but classification fails, missing data (`NaN`) and an error status are recorded, rather than zero.
- *Methodological Note:* This represents a FER model prediction, not a direct measurement of human emotion or psychological intensity.

#### 3. Luminance Contrast ($f_{contrast}$)
Luminance is calculated per pixel from normalized RGB channels ($R, G, B \in [0, 1]$):
$$\large Y = 0.299R + 0.587G + 0.114B$$
Contrast is defined as the population standard deviation ($\sigma$ with $ddof=0$):
$$\large f_{contrast} = \sigma(Y) = \sqrt{\frac{1}{N} \sum_{k=1}^N \left(Y_k - \bar{Y}\right)^2}$$

#### 4. Saturation Variability ($f_{saturation}$)
Images are converted to the HSV color space, with saturation $S$ normalized to $[0, 1]$:
$$\large f_{saturation} = \sigma(S) = \sqrt{\frac{1}{N} \sum_{k=1}^N \left(S_k - \bar{S}\right)^2}$$
- *Methodological Note:* This measures saturation dispersion across the thumbnail canvas, not Hasler-Süsstrunk colorfulness.

---

## 5. Feature Normalization (`05_calculate_scores.py`)

To ensure that features with differing natural scales contribute equitably:

1. **Winsorization & Min-Max Scaling:**
   Applied to the six continuous features: $f_{caps}, f_{trigger}, f_{sensitive}, f_{symbol}, f_{contrast}, f_{saturation}$.
   $$\large x_{scaled} = \frac{\operatorname{clip}(x, q_{0.01}, q_{0.99}) - q_{0.01}}{q_{0.99} - q_{0.01} + 10^{-6}}$$
   Winsorization at the 1st ($q_{0.01}$) and 99th ($q_{0.99}$) percentiles limits the distortion of extreme outliers while bounding scaled values to $[0, 1]$.
2. **Direct Identity Preservation:**
   - $x_{face} = f_{face} \in \{0, 1\}$ (already binary).
   - $x_{expr} = f_{expr} \in [0, 1]$ (already model confidence bounded in $[0, 1]$).

### Empirical Scaling Parameters Table
*(Stored in `scores/scaling_parameters.csv`)*

| Feature Name | Scaled Variable | Method | $q_{0.01}$ | $q_{0.99}$ | Denominator ($q_{0.99} - q_{0.01} + 10^{-6}$) | Raw Mean | Scaled Mean |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `f_caps` | $x_{caps}$ | Winsorize + MinMax | 0.000000 | 1.000000 | 1.000001 | 0.199450 | 0.199449 |
| `f_trigger` | $x_{trigger}$ | Winsorize + MinMax | 0.000000 | 0.333333 | 0.333334 | 0.100148 | 0.298650 |
| `f_sensitive` | $x_{sensitive}$ | Winsorize + MinMax | 0.000000 | 0.250000 | 0.250001 | 0.024238 | 0.094264 |
| `f_symbol` | $x_{symbol}$ | Winsorize + MinMax | 0.000000 | 0.716667 | 0.716668 | 0.090872 | 0.114107 |
| `f_contrast` | $x_{contrast}$ | Winsorize + MinMax | 0.172473 | 0.353491 | 0.181019 | 0.257157 | 0.467179 |
| `f_saturation`| $x_{saturation}$| Winsorize + MinMax | 0.176917 | 0.375223 | 0.198307 | 0.288670 | 0.563524 |
| `f_face` | $x_{face}$ | Direct [0, 1] | 0.000000 | 1.000000 | 1.000000 | 0.935484 | 0.935484 |
| `f_expr` | $x_{expr}$ | Direct [0, 1] | 0.000000 | 0.980000 | 1.000000 | 0.205108 | 0.205108 |

---

## 6.1 Interpretation and sampling limitations

The Click Appeal Score is a researcher-defined descriptive proxy, not a validated measure of psychological response, click probability, or CTR. Views are not CTR; impressions and CTR data would be required to test predictive validity. The 0.4/0.6 weights are researcher-selected baseline assumptions, not empirically optimized. FER outputs are model predictions rather than measured emotions. Face detection can miss or falsely detect faces, particularly for small, stylized, obscured, or edited faces. The lexicons are researcher-defined and may generate false positives. Correlation does not establish causation.

Creator-month averages are exploratory descriptive summaries. The dataset's creator-month cells do not generally meet the five-video threshold; each cell's sample size must be considered, and cells below n=5 must not be described as retained or statistically reliable. The five-video threshold applies to creator-level summaries only.

For thumbnails with no detected face, the current baseline assigns $f_{expr}=0$ while also assigning $f_{face}=0$. This means the baseline composite can penalize a no-face thumbnail through both components. A sensitivity score excluding expression is therefore also supplied in `analysis/sensitivity_analysis.csv` and `scores/video_scores.csv`.

## 6. Composite Score Formulation

Five descriptive score variants are computed for each video:

### 1. Title Score ($X_{title}$)
$$\large X_{title} = \frac{x_{caps} + x_{trigger} + x_{sensitive} + x_{symbol}}{4}$$
- Sample Mean: **0.1753** (Min: 0.0000, Max: 0.7326).

### 2. Thumbnail Score ($X_{thumb}$)
$$\large X_{thumb} = \frac{x_{face} + x_{expr} + x_{contrast} + x_{saturation}}{4}$$
- Sample Mean: **0.5428** (Min: 0.0724, Max: 0.8315).

### 3. Primary Weighted Score ($X_v$) — Researcher Baseline
$$\large X_v = 0.4 X_{title} + 0.6 X_{thumb}$$
- *Weighting Rationale:* The 60% thumbnail / 40% title split is a researcher-selected baseline assumption for this project; it is not justified here as a universal industry or empirically validated weighting.
- *Caveat:* These weights are researcher-selected baselines, not empirically optimized via impression logs.
- Sample Mean: **0.3958** (Min: 0.0734, Max: 0.6569).

### 4. Equal-Weight Alternative Score ($X_{equal}$)
$$\large X_{equal} = \frac{1}{8} \sum_{j=1}^8 x_j = 0.5 X_{title} + 0.5 X_{thumb}$$
- Sample Mean: **0.3591** (Min: 0.0737, Max: 0.6455).

### 5. PCA Alternative Score ($X_{PCA}$)
- Derived from Principal Component Analysis (SVD) on standardized normalized features.
- The first principal component (PC1) explains **21.01%** of total variance.
- Sign orientation: PC1 correlates strongly and positively with $X_{equal}$ ($\rho = +0.7407$).
- Scaled to $[0, 1]$ via min-max:
  $$\large X_{PCA} = \frac{PC_1 - \min(PC_1)}{\max(PC_1) - \min(PC_1)}$$
- Sample Mean: **0.5161** (Min: 0.0000, Max: 1.0000).

#### PC1 Feature Loadings:
- `$x_caps$`: **+0.3368**
- `$x_trigger$`: **+0.2459**
- `$x_sensitive$`: **-0.4543**
- `$x_symbol$`: **+0.3265**
- `$x_face$`: **+0.2632**
- `$x_expr$`: **+0.2069**
- `$x_contrast$`: **+0.3641**
- `$x_saturation$`: **+0.5182**
---

## 7. Complete Worked Example from Real Data

To ensure total transparency, we step through a complete manual verification using actual video record **Row 0**:

### Video Metadata:
- **Video ID:** `oJ6ww2fmuJM`
- **Channel:** `Airrack`
- **Title:** `"I Fooled Strangers With FAKE Celebrities"`
- **Published:** `2024-10-03T16:48:33Z`
- **Views:** `7,103,543` | **Likes:** `104,272` | **Comments:** `2,748`
- **Existing CSV Face Count:** `3`

---

### Step 1: Raw Title Feature Calculation
1. **Tokens:** `['I', 'Fooled', 'Strangers', 'With', 'FAKE', 'Celebrities']` $\rightarrow L = 6$.
2. **Uppercase Tokens ($\ge 2$ letters):**
   - `'FAKE'` has 4 letters, all uppercase $\rightarrow$ count = 1.
   - `'I'` has 1 letter $\rightarrow$ not counted ($< 2$ letters).
   - $$f_{caps} = \frac{1}{6} \approx \mathbf{0.166667}$$
3. **Trigger Words:** No matched trigger words $\rightarrow f_{trigger} = \frac{0}{6} = \mathbf{0.0}$.
4. **Sensitive Words:** No matched sensitive words $\rightarrow f_{sensitive} = \frac{0}{6} = \mathbf{0.0}$.
5. **Symbols/Emojis:** 0 exclamation marks, 0 question marks, 0 emojis $\rightarrow f_{symbol} = \frac{0}{6} = \mathbf{0.0}$.

---

### Step 2: Raw Thumbnail Feature Calculation
1. **Face Detection (MediaPipe):**
   - MediaPipe detected **3 faces** (identical to CSV `face_count = 3`):
     - Face 1: $x=127, y=219, w=224, h=224$, Confidence = 0.844, Area = 50,176 px.
     - Face 2: $x=600, y=166, w=218, h=218$, Confidence = 0.818, Area = 47,524 px.
     - Face 3: $x=858, y=207, w=93, h=93$, Confidence = 0.693, Area = 8,649 px.
   - At least one face detected $\rightarrow \mathbf{f_{face} = 1.0}$.
2. **Expression Classification (FER):**
   - Largest face is Face 1 ($w \times h = 50,176$ px).
   - FER emotion probabilities on $(127, 219, 224, 224)$:
     - Neutral: 0.30, Sad: 0.50, Angry: 0.07, Disgust: 0.04, Happy: 0.01
     - Fear: 0.04, Surprise: 0.05
   - $$f_{expr} = \max(0.05, 0.04) = \mathbf{0.050000}$$
3. **Contrast:**
   - Population standard deviation of luminance:
   - $$\mathbf{f_{contrast} = 0.266458}$$
4. **Saturation:**
   - Population standard deviation of HSV saturation:
   - $$\mathbf{f_{saturation} = 0.354685}$$

---

### Step 3: Normalization
Using empirical parameters from `scores/scaling_parameters.csv`:

1. **$x_{caps}$:**
   $$x_{caps} = \frac{\operatorname{clip}(0.166667, 0.0, 1.0) - 0.0}{1.0 - 0.0 + 10^{-6}} = \frac{0.166667}{1.000001} \approx \mathbf{0.166667}$$
2. **$x_{trigger} = \mathbf{0.0}$**
3. **$x_{sensitive} = \mathbf{0.0}$**
4. **$x_{symbol} = \mathbf{0.0}$**
5. **$x_{face} = \mathbf{1.0}$** (preserved directly)
6. **$x_{expr} = \mathbf{0.05}$** (preserved directly)
7. **$x_{contrast}$:**
   $$q_{0.01} = 0.172473, \quad q_{0.99} = 0.353491, \quad \text{denom} = 0.181019$$
   $$x_{contrast} = \frac{0.266458 - 0.172473}{0.181019} = \frac{0.093985}{0.181019} \approx \mathbf{0.519200}$$
8. **$x_{saturation}$:**
   $$q_{0.01} = 0.176917, \quad q_{0.99} = 0.375223, \quad \text{denom} = 0.198307$$
   $$x_{saturation} = \frac{0.354685 - 0.176917}{0.198307} = \frac{0.177768}{0.198307} \approx \mathbf{0.896427}$$

---

### Step 4: Composite Score Reconstruction
1. **Title Score:**
   $$X_{title} = \frac{0.166667 + 0.0 + 0.0 + 0.0}{4} = \frac{0.166667}{4} = \mathbf{0.041667}$$
2. **Thumbnail Score:**
   $$X_{thumb} = \frac{1.0 + 0.05 + 0.519200 + 0.896427}{4} = \frac{2.465627}{4} = \mathbf{0.616407}$$
3. **Weighted Click Appeal Score ($X_v$):**
   $$X_v = 0.4 \times X_{title} + 0.6 \times X_{thumb}$$
   $$X_v = (0.4 \times 0.041667) + (0.6 \times 0.616407)$$
   $$X_v = 0.016667 + 0.369844 = \mathbf{0.386511}$$
4. **Equal-Weight Score ($X_{equal}$):**
   $$X_{equal} = \frac{0.166667 + 0 + 0 + 0 + 1.0 + 0.05 + 0.519200 + 0.896427}{8} = \frac{2.632294}{8} = \mathbf{0.329037}$$
5. **PCA Score ($X_{PCA}$):**
   $$\mathbf{X_{PCA} = 0.584273}$$

*Validation Check:* In `scores/video_scores.csv`, row 0 records:
`score_title: 0.041667`, `score_thumbnail: 0.616407`, `score_weighted: 0.386511`, `score_equal_weight: 0.329037`.  
The manual arithmetic matches the pipeline outputs down to the 6th decimal place.

---

## 8. Manual Verification Checklist

Researchers and auditors can verify every component using this five-step checklist:

- [x] **Checklist Item 1: Tokenization Count ($L$)**
  - Count whitespace-separated tokens in title. Check that $L = \max(1, \text{len}(\text{tokens}))$.
- [x] **Checklist Item 2: Uppercase Tokens Count**
  - Verify that each counted uppercase token contains $\ge 2$ letters and that all letters are uppercase. Verify that punctuation is stripped.
- [x] **Checklist Item 3: Face Bounding Box & Area Ranking**
  - Confirm that face bounding boxes lie within image dimensions ($W \times H$).
  - For thumbnails with multiple faces, confirm that $A_i = w_i \times h_i$ correctly identifies the largest face for emotion analysis.
- [x] **Checklist Item 4: Winsorization Boundary Check**
  - Check that values smaller than $q_{0.01}$ are clamped to $q_{0.01}$ and values greater than $q_{0.99}$ are clamped to $q_{0.99}$. Check that the scaled value strictly falls in $[0, 1]$.
- [x] **Checklist Item 5: Score Reconstructibility Equation**
  - Plug $X_{title}$ and $X_{thumb}$ into $0.4 X_{title} + 0.6 X_{thumb}$ and confirm that the result matches `score_weighted` within $10^{-5}$ tolerance.

---

## 9. Creator-Level Aggregation & LOCO Benchmarking

### 9.1 Grouping and Thresholding
- **Channel Aggregation:** All 38 channels are aggregated.
- **Threshold Criterion:** Channels with $\ge 5$ valid videos are retained for statistical ranking. Exactly 37 channels meet this criterion (each with 5 videos). One channel (`Furious Pete`, 1 video) is retained but flagged as below the 5-video cohort threshold.
- **Monthly Cohorts:** Videos are grouped by channel and publication month, generating 182 creator-month observations. Every group is below n=5 (maximum 3), so these are exploratory descriptive summaries, not threshold-qualified cohorts (`scores/creator_monthly_scores.csv`).

### 9.2 Leave-One-Creator-Out (LOCO) Benchmark
To establish an unbiased benchmark, each creator $c$'s performance is compared against the niche average computed **without** that creator's videos:
$$\large \mu_{\text{LOCO}, c} = \frac{1}{N - N_c} \sum_{i \notin c} X_{v, i}$$
$$\large \text{Relative Lift} = \bar{X}_{v, c} - \mu_{\text{LOCO}, c}$$

### 9.3 Creator Rankings (Top & Bottom Cohorts)

Creator-level means are descriptive. The five-video threshold is met by 37 creators; one creator has a single video and is retained but flagged. Values below are recalculated after the title-feature correction. Full results are in `scores/creator_summary_scores.csv`.

| Rank | Channel Name | Videos | Mean $X_v$ | Std Dev | Mean $X_{title}$ | Mean $X_{thumb}$ | Niche Mean | LOCO Mean | Relative Lift |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Wake'N'Bite | 5 | 0.540278 | 0.076427 | 0.429205 | 0.614326 | 0.395833 | 0.391843 | +0.148434 |
| 2 | Max vs Food | 5 | 0.524571 | 0.088965 | 0.366949 | 0.629651 | 0.395833 | 0.392277 | +0.132294 |
| 3 | BeardMeatsFood | 5 | 0.490120 | 0.071558 | 0.380032 | 0.563513 | 0.395833 | 0.393229 | +0.096891 |
| 4 | Zermatt Neo | 5 | 0.470884 | 0.105659 | 0.256730 | 0.613653 | 0.395833 | 0.393760 | +0.077123 |
| 5 | Foodie We | 5 | 0.469269 | 0.073178 | 0.265769 | 0.604935 | 0.395833 | 0.393805 | +0.075464 |
| 35 | How Ridiculous | 5 | 0.323740 | 0.144999 | 0.150775 | 0.439050 | 0.395833 | 0.397825 | -0.074085 |
| 36 | Matt Stonie | 5 | 0.306412 | 0.115998 | 0.087401 | 0.452420 | 0.395833 | 0.398304 | -0.091891 |
| 37 | The Bucket List Family | 5 | 0.257643 | 0.113196 | 0.098541 | 0.363712 | 0.395833 | 0.399651 | -0.142008 |
| 38 | FreakEating | 5 | 0.225885 | 0.152889 | 0.082342 | 0.321580 | 0.395833 | 0.400528 | -0.174643 |
---

## 10. Key Descriptive Findings & Statistical Analysis

1. **Descriptive component averages:**
   - The sample mean thumbnail component score is 0.5428 and the sample mean title component score is 0.1753. Because the component features have different constructions and distributions, this difference should not be interpreted as evidence of greater causal influence or predictive power.
2. **Face Detection Discrepancy Analysis:**
   - MediaPipe detected faces in 93.5% of thumbnails (174/186), whereas the existing CSV labeled faces in 83.3% (155/186).
   - Exact count agreement was **59.7%** (111/186). The disagreement requires case-by-case visual review; the available metadata alone does not establish which detector or source label is correct.
3. **Exploratory Correlation with Video Engagement:**
   - Spearman rank correlation with $\log_{10}(\text{views})$: $r_s = -0.01$ (effectively zero across this cross-sectional sample of top channels).
   - Spearman rank correlation with $\log_{10}(\text{likes})$: $r_s = -0.04$.
   - Spearman rank correlation with $\log_{10}(\text{comments})$: $r_s = +0.02$.
   - **Crucial Scientific Interpretation:** View counts are driven by channel subscriber base, recommendation velocity, and lifetime age, whereas Click Appeal measures frontend packaging. Without impression denominator data, views cannot be interpreted as CTR.

---

## 11. Assumptions, Scientific Limitations, & Methodological Caveats

1. **Views Are NOT Click-Through Rate (CTR):**
   - Click-Through Rate is defined as $\text{CTR} = \frac{\text{Clicks}}{\text{Impressions}}$. The dataset provides lifetime view counts, not platform impressions. A video with 10 million views may have had a 4% CTR on 250 million impressions, while a video with 100,000 views may have had a 12% CTR on 830,000 impressions.
2. **Absence of Causal Claims:**
   - Observational correlations cannot establish that changing title capitalization or thumbnail contrast causes higher viewership. Algorithmic distribution, viewer retention, watch time, and channel authority are major confounding factors.
3. **Researcher-Defined Baseline Weights:**
   - The $0.4 / 0.6$ weighting split is an operational hypothesis. Researchers should evaluate the equal-weight ($X_{equal}$) and PCA ($X_{PCA}$) alternatives provided in `video_scores.csv`.
4. **Model Confidence vs. Psychological Affect:**
   - FER outputs indicate neural network confidence in categorized facial landmark patterns, not internal psychological states.

---

## 12. Reproducibility Guide

To reproduce this pipeline from scratch:

```bash
# 1. Clone/navigate to project root
cd ClickAppeal_New

# 2. Set up Python 3.12 virtual environment
py -3.12 -m venv .venv

# 3. Activate environment & install requirements
.\.venv\Scripts\pip install -r requirements.txt

# 4. Execute master pipeline orchestrator
.\.venv\Scripts\python run_pipeline.py
```

All intermediate outputs, features, visual audit images, and statistical figures will be deterministically reconstructed.
