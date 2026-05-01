# 🏥 30-Day Hospital Readmission Risk Prediction
### End-to-End Machine Learning Project | XGBoost + SHAP + Power BI

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-orange)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-green)
![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-yellow?logo=powerbi)
![Dataset](https://img.shields.io/badge/Dataset-101%2C766%20patients-lightblue)
![Cost](https://img.shields.io/badge/Total%20Cost-₹0-brightgreen)

---

## 📌 Problem Statement

Hospital readmissions within 30 days represent one of the most expensive and preventable problems in modern healthcare. In the United States alone, 30-day readmissions cost the healthcare system over **$26 billion annually**. The Centers for Medicare and Medicaid Services (CMS) introduced the **Hospital Readmissions Reduction Program (HRRP)** which financially penalises hospitals that have excess 30-day readmission rates for conditions including diabetes, heart failure, and pneumonia.

Diabetic patients are among the highest-risk groups — they often have multiple comorbidities, complex medication regimens, and frequent prior hospitalisations that make them prone to returning to the hospital within 30 days of discharge.

The core challenge is not just predicting who will be readmitted — **any model can do that with enough data**. The real challenge is explaining *why* a specific patient is flagged as high risk, so that doctors, nurses, and hospital administrators can take targeted preventive action before discharge.

> **This project solves both problems:** a high-performance XGBoost classifier that predicts 30-day readmission risk, combined with SHAP explainability that tells clinicians exactly which factors are driving each individual patient's risk score.

---

## 🎯 Project Objectives

1. Build an end-to-end ML pipeline on real hospital data — from raw messy CSV to production-ready model
2. Engineer clinically meaningful features from domain knowledge, not just statistical tricks
3. Handle class imbalance (only 9% positive class) using SMOTE without data leakage
4. Make every prediction explainable using SHAP — not just a black-box risk score
5. Deliver findings through an interactive Power BI dashboard usable by non-technical clinical staff

---

## 📊 Dataset Overview

| Detail | Value |
|---|---|
| Name | Diabetes 130-US Hospitals Dataset |
| Source | UCI Machine Learning Repository / Kaggle |
| Original size | 101,766 patient encounters |
| Final size after cleaning | 69,990 unique patients |
| Time period | 1999–2008 |
| Hospitals covered | 130 US hospitals |
| Features | 50 raw → 24 engineered |
| Target | `readmitted_30` (1 = readmitted within 30 days, 0 = not) |
| Class imbalance | ~9% positive, ~91% negative |

---

## 🗂️ Repository Structure

```
healthcare-readmission/
│
├── Healthcare_Readmission_FINAL.ipynb    # Complete project — all days in one notebook
├── README.md                             # This file
├── requirements.txt                      # Python dependencies
│
├── data/
│   └── sample_data.csv                   # 500-row sample (full dataset on Kaggle)
│
└── outputs/
    ├── eda_charts/                        # 10 EDA visualisations
    ├── model_charts/                      # Confusion matrix, ROC curve, feature importance
    └── shap_charts/                       # 5 SHAP explainability charts
```

---

## 📅 Day-by-Day Project Workflow

---

### 📗 Day 1 — Exploratory Data Analysis (EDA)

**Goal:** Understand the raw dataset before touching it. Build intuition about what drives readmission before writing a single line of ML code.

**Description:**

The raw dataset was loaded — 101,766 rows across 50 columns. The first task was understanding the target variable. The `readmitted` column has three values: `<30` (readmitted within 30 days), `>30` (readmitted after 30 days), and `NO` (not readmitted). For binary classification, we define `<30` as the positive class.

10 charts were produced covering every dimension of the data:

- **Target distribution** — Only 9% positive class. Standard accuracy is meaningless here. AUC-ROC is the correct evaluation metric.
- **Missing value analysis** — Three columns have extreme missingness: `weight` (97%), `medical_specialty` (49%), `payer_code` (40%). The dataset also encodes missing values as `"?"` strings rather than `NaN` — a critical discovery for preprocessing.
- **Numeric feature distributions** — `number_inpatient`, `number_outpatient`, `number_emergency` are right-skewed with most patients at zero but a small high-utilisation group at much higher values. This high-utilisation group would prove to be the highest-risk segment.
- **Readmission by age** — The 70–80 age bracket has the highest readmission rate (11%) AND the highest patient volume. This cohort is the highest-priority target for clinical intervention.
- **Prior inpatient visits vs readmission** — The strongest single finding in EDA: patients with 5+ prior inpatient visits have 3x the 30-day readmission rate compared to first-time patients. This variable would later be confirmed as a top SHAP feature.
- **Medication analysis** — Patients on diabetes medication have higher readmission rates, reflecting that medicated patients have more severe disease.
- **Correlation heatmap** — `number_inpatient` has the highest linear correlation with the target among all numeric features.

**Key EDA finding for interviews:**
> *"Before touching the model, EDA told me that prior inpatient visits would be the strongest predictor — and SHAP later confirmed it. That alignment between domain intuition and model output gives me confidence in the results."*

---

### 📙 Day 2 — Data Cleaning & Preprocessing

**Goal:** Clean the raw data systematically, with every decision documented and justifiable to a client or interviewer.

**Description:**

**Step 1 — Replace "?" with NaN**
The dataset uses `"?"` as a missing value indicator. pandas cannot detect this without explicit conversion. Over 98,000 `"?"` values were found across the dataset.

**Step 2 — Drop high-missing columns (>40% threshold)**
`weight` (97% missing), `medical_specialty` (49%), and `payer_code` (40%) were dropped. Imputing columns with this level of missingness would introduce more noise than signal — these columns cannot be reliably reconstructed.

**Step 3 — Remove clinically invalid records**
Patients with discharge disposition IDs 11, 13, 14, 19, 20, 21 (deceased, hospice, long-term care) were removed. These patients **cannot** be readmitted within 30 days by definition. Keeping them would corrupt the target variable — the model would learn that dead patients are "not readmitted" which is trivially true and clinically meaningless. Removed ~2,423 records.

**Step 4 — Deduplicate to one encounter per patient**
The dataset contains multiple encounters for the same `patient_nbr`. Training on all encounters means the model sees the same person multiple times — a form of data leakage where the model can "memorise" patient patterns rather than generalising. Kept only the first encounter per patient. This simulates the real-world scenario where we predict risk at a patient's first presentation.

**Step 5 — Drop ID columns**
`encounter_id` and `patient_nbr` are arbitrary identifiers with zero predictive value.

**Step 6 — Create binary target**
`readmitted_30` = 1 if `readmitted == '<30'`, else 0. The 30-day window is the CMS HRRP standard. Final positive rate: **8.98%**.

**Step 7 — Impute remaining missing**
Numeric columns → median (robust to outliers). Categorical columns → mode (most frequent). Zero missing values after imputation.

**Step 8 — Remove zero-variance columns**
`examide`, `citoglipton`, and `glimepiride-pioglitazone` had only one unique value across all rows — they carry no information.

**Output:** 69,990 × 26 columns. Saved as `cleaned_diabetic_data.csv`.

---

### 📘 Day 3 — Feature Engineering

**Goal:** Transform raw columns into features that carry clinical meaning. This is where business and domain thinking — not just coding — determines model quality.

**Description:**

**Age encoding**
The `age` column contains string ranges like `[70-80)`. Mapped to numeric midpoints (5, 15, 25... 95). This preserves the ordinal relationship (75 > 45) which alphabetic label encoding would destroy.

**4 Domain-Driven Features Created:**

| Feature | Formula | Clinical Reason |
|---|---|---|
| `total_visits` | outpatient + inpatient + emergency visits | Overall healthcare utilisation predicts readmission better than any single visit type |
| `high_risk_age` | 1 if age ≥ 70 else 0 | EDA confirmed 70+ cohort has highest volume and highest readmission rate |
| `meds_changed` | 1 if any drug went Up or Down during admission | Medication changes signal treatment instability and disease severity |
| `num_meds_active` | Count of drugs not equal to "No" | Polypharmacy (many active drugs) is a known readmission risk factor in clinical literature |

**ICD Diagnosis Grouping**
`diag_1`, `diag_2`, `diag_3` contained 700+ unique ICD-9 codes. One-hot encoding would create 700+ binary columns — the curse of dimensionality. Instead, each code was mapped to one of 9 standard WHO disease chapters (Circulatory, Respiratory, Digestive, Diabetes, Injury, Musculoskeletal, Genitourinary, Neoplasms, Other). Label encoded rather than one-hot to keep the feature matrix lean — XGBoost handles label-encoded integers natively.

**Train/Test Split**
80/20 stratified split. `stratify=y` is critical — without it, random chance could put most positive-class patients in one split, making evaluation invalid.

**StandardScaler**
Fit on training data only, then transform both train and test. Fitting on test data would constitute data leakage.

**Final feature matrix:** 69,990 × 24 columns.

---

### 📕 Day 4 — XGBoost Model + SMOTE

**Goal:** Train a high-performance classifier that handles class imbalance correctly and produces a model ready for SHAP explainability.

**Description:**

**Why XGBoost:**
Tree-based models like XGBoost consistently outperform linear models on tabular healthcare data. Additional reasons: no feature scaling required, built-in handling of missing values, natively compatible with SHAP's TreeExplainer, and widely used in consulting and healthcare analytics engagements.

**SMOTE — Class Imbalance Handling**
With 9% positive class, a naive model learns to predict "Not Readmitted" almost always and achieves 91% accuracy while being clinically useless. SMOTE (Synthetic Minority Oversampling Technique) creates synthetic positive-class samples by interpolating between real positive samples in the feature space.

**The most important rule:** SMOTE was applied **only to training data**. The test set remained entirely real and untouched. This is a common mistake — applying SMOTE to test data artificially inflates recall and AUC metrics, making the model appear better than it is.

Before SMOTE: ~51,000 class 0 vs ~5,000 class 1 (10:1 imbalance)
After SMOTE: ~51,000 class 0 vs ~51,000 class 1 (balanced)

**Model Configuration:**
```python
XGBClassifier(
    n_estimators     = 300,   # number of trees
    max_depth        = 5,     # controls complexity
    learning_rate    = 0.1,   # shrinkage per step
    subsample        = 0.8,   # 80% rows per tree
    colsample_bytree = 0.8,   # 80% features per tree
    eval_metric      = 'auc'  # optimise for AUC not accuracy
)
```

**Results:**

| Metric | Value |
|---|---|
| AUC-ROC | ~0.74 |
| Baseline AUC (random) | 0.50 |
| Improvement over baseline | +48% |

**Why AUC and not Accuracy:**
With 91% negative class, a model predicting "Not Readmitted" always achieves 91% accuracy. AUC-ROC measures the model's ability to rank positive cases above negative cases — independent of the decision threshold. It is the correct metric for imbalanced binary classification.

**Outputs saved:** `models/xgb_model.pkl`, `predictions.csv` with risk bands (Low/Medium/High), 5 evaluation charts.

---

### 📒 Day 5 — SHAP Explainability

**Goal:** Transform a black-box risk score into a clinically actionable explanation tool. Make every prediction interpretable to a doctor.

**Description:**

SHAP (SHapley Additive exPlanations) is grounded in cooperative game theory. Each feature's SHAP value represents its average marginal contribution to the prediction across all possible orderings of features — a mathematically rigorous measure of feature importance.

**SHAP vs Built-in Feature Importance:**
XGBoost's built-in importance shows how often a feature was used to split nodes — it doesn't show direction or magnitude of impact per patient. SHAP gives each feature a signed contribution per patient:
- Positive SHAP = this feature pushed the prediction toward readmission
- Negative SHAP = this feature pushed the prediction away from readmission

**5 Charts Produced:**

**Chart 1 — Bar Plot (Global Importance)**
Mean absolute SHAP value per feature across all 13,998 test patients. Shows which features matter most at the population level.

**Chart 2 — Beeswarm Plot (Direction + Magnitude)**
Each dot = one patient. X-position = SHAP value. Colour = feature value (red = high, blue = low). This chart simultaneously shows importance, direction, and the distribution of impact across the population. It is the most information-dense and visually compelling chart in the project.

**Chart 3 — Waterfall Plot (High-Risk Patient)**
The clinical gold of the project. Shows exactly why one specific high-risk patient was flagged — feature by feature. The baseline E[f(x)] is the average predicted probability across training patients. Each bar shows how that patient's specific feature value moved the prediction up or down. A clinician can read this chart and immediately know which factors to address before discharge.

**Chart 4 — Waterfall Plot (Low-Risk Patient)**
Contrast to Chart 3. Demonstrates complete understanding of the model's decision space — not just why patients are flagged high risk, but why others are considered safe.

**Chart 5 — Dependence Plot**
Shows how the SHAP value for `number_inpatient` changes as its value increases — confirming the monotonic relationship identified in EDA. Coloured by `number_emergency` to reveal interaction effects between prior visit types.

**Important point:**
> *"I used SHAP's TreeExplainer which computes exact Shapley values using the tree structure — no approximations needed. This gives mathematically grounded, patient-level explanations that a hospital discharge team can act on immediately."*

---

### 📓 Day 6 — Power BI Dashboard

**Goal:** Deliver model outputs to non-technical stakeholders through an interactive clinical-grade dashboard.

**Description:**

Two-page dashboard built in Power BI Desktop using `predictions.csv` and `shap_summary.csv`.

**Page 1 — Executive Overview**
Designed for hospital administrators and clinical directors. Answers: *"Who is at risk and what does the risk profile of our patient population look like?"*

- 4 KPI cards: Average Risk Score, Total Patients, Actual Readmissions, High Risk Patients
- Bar chart: Readmission rate by patient age group (confirms 70–80 peak from EDA)
- Donut chart: Patient risk band distribution (Low / Medium / High)
- Line chart: Average risk score vs number of medications
- 3 interactive slicers: Risk Band, Diabetes Medication, Age Range

**Page 2 — SHAP Model Insights**
Designed for data-literate clinicians. Answers: *"Why is the model flagging these patients — and which factors should we target clinically?"*

- SHAP feature importance bar chart
- Embedded beeswarm and waterfall images
- High-risk patient table with conditional risk colouring (red >0.7, amber 0.4–0.7, green <0.4)

---

## 📈 Key Results

| What | Result |
|---|---|
| Model | XGBoost + SMOTE |
| AUC-ROC | ~0.74 |
| Baseline AUC | 0.50 |
| Top readmission driver | `num_meds_active` (active medications) |
| Second driver | `number_inpatient` (prior inpatient visits) |
| Third driver | `total_visits` (total healthcare utilisation) |
| Explainability | Per-patient SHAP waterfall explanations |
| Dashboard | 2-page interactive Power BI |
| Total cost | ₹0 |

---

## 💡 Business Recommendations

Based on SHAP analysis, three clinical interventions are recommended for patients with predicted probability > 0.6:

1. **Medication review before discharge** — Patients with 15+ active medications should receive a pharmacist consultation. `num_meds_active` is the top SHAP feature.

2. **Enhanced follow-up for frequent admitters** — Patients with 3+ prior inpatient visits should be enrolled in a post-discharge monitoring programme. `number_inpatient` is the second strongest predictor.

3. **Age-targeted discharge planning** — The 70–80 age bracket has the highest readmission rate and highest patient volume. Discharge planning resources should be prioritised for this cohort.

---

## ⚙️ Tech Stack

| Tool | Purpose | Cost |
|---|---|---|
| Python 3.13 | Core language | Free |
| pandas, numpy | Data manipulation | Free |
| matplotlib, seaborn | Visualisations | Free |
| scikit-learn | ML pipeline, metrics, scaling | Free |
| XGBoost | Classification model | Free |
| imbalanced-learn | SMOTE oversampling | Free |
| SHAP | Model explainability | Free |
| Power BI Desktop | Interactive dashboard | Free |
| Kaggle | Dataset source | Free |

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/sandeepsingh9871/30-Day-Hospital-Readmission-Risk-Prediction.git
cd healthcare-readmission
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the full dataset**
Download `diabetic_data.csv` from [Kaggle]((https://www.kaggle.com/competitions/widsdatathon2021/discussion/215380)) and place in the project root.

**4. Run the notebook**
```bash
jupyter notebook healthcare_readmission.ipynb
```
Run all cells top to bottom. Do not skip cells. Total runtime: ~10 minutes.

---

## 📋 Files to Upload on GitHub

```
✅ healthcare_readmission.ipynb   — main notebook
✅ README.md                            — this file
✅ requirements.txt                     — dependencies
✅ data/sample_data.csv                 — 500-row sample only
✅ outputs/shap_charts/                 — all 5 SHAP charts
✅ outputs/model_charts/               — confusion matrix, ROC, feature importance
✅ outputs/eda_charts/06_inpatient_visits_readmission.png
✅ outputs/eda_charts/07_correlation_heatmap.png

❌ diabetic_data.csv                   — too large, link to Kaggle instead
❌ cleaned_diabetic_data.csv           — generated file, not needed
❌ data/processed/*.pkl                — binary files, no use on GitHub
❌ models/xgb_model.pkl               — too large for GitHub
❌ predictions.csv / shap_summary.csv — generated files
```

---

## 📧 Contact

**Sandeep**
- LinkedIn: [(https://www.linkedin.com/in/sandeep-singh-aaa1271b7/)]
- Email: [sandeepsinghss9871@gmail.com]
- GitHub: [(https://github.com/sandeepsingh9871)]

---

*Built as part of a data analytics portfolio targeting healthcare analytics and management consulting roles (Deloitte, PwC, EY, KPMG, Accenture, ZS Associates).*
