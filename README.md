# ⚡ NEC ML Pipeline – Smart Power Plant Selection

End-to-end machine learning pipeline to predict electricity generation costs and select the optimal power plant for each demand scenario using decision-focused evaluation.

**Institution:** Keele Business School  
**Module:** MAN-40389 – Advanced Data Analytics and Machine Learning  
**Assessment:** Group Coursework (60%)

---

## 📌 Executive Summary

This project delivers an automated ML pipeline that supports operational decision-making in electricity generation.  
Rather than optimizing prediction accuracy only, the pipeline is evaluated by its ability to select the lowest-cost plant for each demand scenario.


---

## 🎯 Problem Definition

Electricity providers must decide which power plant to dispatch for a given demand while minimising cost.

### Given
- Demand characteristics (environmental & contextual features)
- Power plant characteristics (technical & operational features)
- Historical generation costs

### Objectives
1. Predict generation cost (USD/MWh)
2. Rank plants per demand scenario
3. Select the lowest-cost plant
4. Minimise financial loss due to incorrect selection

---

## 🧠 Data Overview

### Demand Data
- 500 demand scenarios  
- Numerical demand features `DF1–DF12`  
- Categorical fields: `DF_region`, `DF_daytype`

### Power Plant Data
- 64 plants  
- Numerical plant features `PF1–PF18`  
- Categorical fields: `Plant Type`, `Region`

### Generation Cost Data
- Target: `Cost_USD_per_MWh` for each (Demand ID, Plant ID)

---

## 🔄 Pipeline Overview (main.py)

The entire pipeline runs from a single entry point: `main.py`

1. **Ingestion** – load 3 datasets and merge into a master table  
2. **Validation** – schema checks, duplicates, target missing warnings, coverage warnings  
3. **Preprocessing**
   - Numeric: median imputation + standard scaling  
   - Categorical: imputation + one-hot encoding  
4. **Held-out split (group-aware)** – split by **Demand ID**  
5. **Baseline** – DummyRegressor evaluated on held-out  
6. **Untuned model** – train on train set, evaluate on held-out  
7. **Hyperparameter tuning** – **GridSearchCV** over a full **Pipeline(preprocess + model)**  
   - CV splitter: **LeaveOneGroupOut (LOGO)** by Demand ID  
   - Scoring: **RMSE Selection Error** (custom scorer)  
8. **Final evaluation** – tuned pipeline evaluated on held-out  
9. **Artifacts saved** – stored in a unique run folder under `artifacts/`

---

## 📊 Evaluation Strategy

### Regression Metric (secondary)
- **RMSE (regression)**: RMSE between true and predicted costs (USD/MWh)

### Decision-Focused Metric (primary NEC KPI)
For each Demand ID:

- Choose plant:  `argmin(predicted_cost)`  
- True best plant: `argmin(true_cost)`  
- **Selection Error** = `true_cost(chosen) - true_cost(best)` 

Report:
- **Mean Selection Error**
- **RMSE Selection Error** (used for tuning)

---

## 🚀 Quick Start

### Prerequisites
```
Python 3.8+
Minimum 8GB RAM (16GB recommended for LOGO tuning)
```

### Setup
```bash
# 1) Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

---

## ▶️ Running the Pipeline
```bash
# Execute the full pipeline
python main.py
```
---

## 📂 Project Structure

```
NEC_ML_Pipeline_Group1-main/
├── artifacts/
│   ├── preprocessor.pkl
│   └── technical_summary_report.txt
│   └── trained_model.pkl
├── config/
│   └── config.yaml
├── data/
│   ├── demand.csv
│   ├── plants.csv
│   └── generation_costs.csv
├── src/
│   └── nec_ml_pipeline/
│       ├── __init__.py
│       ├── ingestion.py
│       ├── validation.py
│       ├── preprocessing.py
│       ├── models.py
│       ├── evaluation.py
│       └── utils.py
├── artifacts/
│   └── run_YYYYMMDD_HHMMSS/   # generated per run
├── main.py
├── requirements.txt
└── README.md

---

## 💾 Output Files

To ensure reproducibility, all outputs are automatically saved to the `artifacts/` directory after execution:

```text
artifacts/
  run_YYYYMMDD_HHMMSS/
    config_used.yaml
    model_pipeline.joblib
    cv_results.csv
    heldout_results.json
    selection_table.csv
    selection_table_untuned.csv
    technical_summary_report.txt            

## ⚙️ Environment Details

### Core Libraries
- pandas
- numpy
- scikit-learn
- matplotlib
- pyyaml
- joblib

---
## ✅ Key Results (Held-out by Demand ID)

Evaluation used a **group-aware held-out split**: 80% train demands (400) / 20% test demands (100).  
Hyperparameter optimisation used **Leave-One-Group-Out (LOGO)** on the training demands and optimised **RMSE Selection Error**.

### Best Hyperparameters (Random Forest)
- `n_estimators`: 100  
- `max_depth`: 10  
- `min_samples_split`: 2  
Best CV RMSE Selection Error (LOGO, train only): **3.7321**

### Held-out Performance (Primary NEC KPI: Selection Error)
| Model | RMSE Selection Error ↓ | Mean Selection Error ↓ | RMSE (Regression) ↓ |
|------|--------------------------|-------------------------|----------------------|
| Baseline (Dummy) | **26.0302** | 25.0011 | 14.7053 |
| Untuned RF | **4.9440** | 3.1214 | 11.9240 |
| Tuned RF (LOGO optimised) | **4.7029** | 3.1054 | 12.0143 |

**Impact:** Tuned RF reduces RMSE Selection Error by ~**82%** vs baseline  
(26.0302 → 4.7029), demonstrating strong decision-focused improvement.

