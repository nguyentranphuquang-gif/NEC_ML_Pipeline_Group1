# ⚡ NEC ML Pipeline – Smart Power Plant Selection

End-to-end machine learning pipeline for predicting electricity generation costs and optimally selecting power plants for different demand scenarios using data-driven decision making.

**Institution:** Keele University Business School  
**Module:** MAN-40389 – Advanced Data Analytics and Machine Learning  
**Assessment:** Group Coursework (60%)

---

## 📌 Executive Summary

This project implements an automated machine learning system that supports operational decision-making in energy generation.  
Instead of focusing only on prediction accuracy, the pipeline evaluates models based on their ability to select the most cost-effective power plant for each demand scenario.

### Key Highlights
- **Dataset:** 26,560 observations (415 demand scenarios × 64 power plants)
- **Learning Task:** Supervised regression (cost prediction)
- **Decision Task:** Lowest-cost plant selection per demand
- **Final Model:** Tuned Random Forest
- **Outcome:** 4.0% reduction in selection error compared to baseline

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
- 415 demand scenarios
- Numerical demand features (DF1–DF12)
- Categorical attributes (e.g. region, demand type)

### Power Plant Data
- 64 power plants
- Numerical plant features (PF1–PF18)
- Metadata (plant type, region)

### Generation Cost Data
- Cost (USD/MWh) for every demand–plant combination
- Used as the supervised learning target

---

## 🔄 Machine Learning Pipeline

The pipeline is executed end-to-end from a single entry point (`main.py`):

1. Data ingestion – load and merge datasets
2. Validation – schema and integrity checks
3. Preprocessing  
   - Numerical: median imputation + scaling  
   - Categorical: one-hot encoding
4. Baseline model – Dummy regressor
5. Model training – Random Forest
6. Cross-validation – GroupKFold (by Demand ID)
7. Hyperparameter tuning – GridSearchCV
8. Evaluation – RMSE + decision-based metrics
9. Persistence – models, reports, plots

---

## 📊 Evaluation Strategy

### Regression Metrics
- RMSE
- R²

### Decision-Focused Metric (Key Contribution)
**Selection Error** evaluates whether the model selects the same plant as the true minimum-cost plant for each demand scenario.

This directly measures real-world economic impact rather than prediction accuracy alone.

---

## 🚀 Quick Start

### Prerequisites
```
Python 3.8+
Minimum 8GB RAM (16GB recommended for full tuning)
```

### Installation
```bash
git clone <repository-url>
cd nec-ml-pipeline

python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

---

## ▶️ Running the Pipeline

```bash
# Full pipeline with hyperparameter tuning (~15–20 mins)
python main.py

# Quick mode (~5–8 mins)
python main.py --quick

# Baseline only (~2–3 mins)
python main.py --no-tune

# Minimal output
python main.py --quick --quiet
```

---

## 📈 Expected Console Output

```
====================================================
NEC ML PIPELINE – COMPLETE EXECUTION
====================================================

[1/8] Data Loading
[2/8] Preprocessing
[3/8] Baseline Model
[4/8] Baseline Evaluation
[5/8] Hyperparameter Tuning
[6/8] Tuned Model Evaluation
[7/8] Model Comparison
[8/8] Results Saved

Final Selection Error Improvement: 4.0%
====================================================
```

---

## 📂 Project Structure

```
nec-ml-pipeline/
├── data/
│   ├── raw/
│   └── processed/
├── src/
├── tests/
├── results/
│   ├── evaluation_reports/
│   ├── selection_tables/
│   └── plots/
├── models/
├── main.py
├── requirements.txt
└── README.md
```

---

## 🧪 Testing

```bash
python -m tests.test_data_ingestion
python -m tests.test_preprocessing
python -m tests.test_models
python -m tests.test_evaluation
python -m tests.test_tuning

pytest tests/
```

---

## 🧾 Key Results

| Metric | Baseline RF | Tuned RF | Improvement |
|------|------------|----------|------------|
| RMSE | 13.01 | 12.90 | 0.9% |
| R² | 0.332 | 0.344 | 3.5% |
| Selection Error | 60.24% | 57.83% | 4.0% |
| Correct Selections | 33 / 83 | 35 / 83 | +2 |

### Tuned Model Parameters
- n_estimators: 200
- max_depth: None
- max_features: sqrt
- min_samples_split: 5
- min_samples_leaf: 2

---

## 💾 Output Files

```
results/
├── evaluation_reports/
├── selection_tables/
└── plots/

models/
└── tuned_rf_<timestamp>.pkl
```

This ensures full reproducibility of results.

---

## ⚙️ Environment Details

### Core Libraries
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

### Tested On
- Windows 10 / 11
- macOS 12+
- Ubuntu 20.04+

---

## 🛠 Troubleshooting

**Module errors**
```bash
pip install -r requirements.txt
```

**Memory issues**
```bash
python main.py --quick
```

**Slow execution**
```bash
python main.py --no-tune
```

**Import errors**
```bash
cd nec-ml-pipeline
python main.py
```

---

## 📚 Documentation

- Clear docstrings across all modules
- Consistent function naming
- Type hints for key components

Supplementary coursework documents:
- Assessment brief
- Technical summary
- Presentation slides

---

## 👥 Team Contributions

- Data ingestion & validation
- Feature engineering & preprocessing
- Model training & tuning
- Custom decision-based evaluation
- Integration & testing

All components were fully integrated and jointly validated.

---

## 📖 Citation

```bibtex
@software{nec_ml_pipeline_2025,
  title = {NEC ML Pipeline: Smart Power Plant Selection},
  year = {2025},
  institution = {Keele University Business School},
  course = {MAN-40389}
}
```
