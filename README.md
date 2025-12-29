# NEC_ML_Pipeline_Group1
NEC Plant Selection ML Pipeline - Group 1 (MAN-40389)

📌 Project Overview

This project is an end-to-end Machine Learning (ML) pipeline designed to support cost-efficient electricity generation planning.
The system uses historical electricity demand, power plant characteristics, and generation costs to train and evaluate regression models that help determine optimal power plant usage.

The project follows industry-style ML pipeline practices, including data ingestion, validation, preprocessing, model training, evaluation, and artifact generation.

🎯 Problem Statement

Electricity providers must meet varying demand levels while minimizing operational costs.
Different power plants have different capacities, efficiencies, and generation costs.

Objective:

Build a machine learning pipeline that analyzes demand and power plant data to model generation cost patterns and support efficient power plant selection.

🧠 Machine Learning Approach

ML Type: Supervised Learning

Task: Regression

Target Variable: Electricity generation cost

Baseline Model: Dummy Regressor

Evaluation Metric: Mean Squared Error (MSE)

Validation Strategy: Group K-Fold Cross Validation

🏗️ Project Architecture

The project is organized as a modular ML pipeline with clear separation of responsibilities.

NEC_ML_Pipeline_Group1
│
├── data/
│   ├── demand.csv              # Electricity demand data
│   ├── plants.csv              # Power plant characteristics
│   └── generation_costs.csv    # Historical generation costs
│
├── src/
│   ├── ingestion.py            # Data loading & merging
│   ├── validation.py           # Data schema validation
│   ├── preprocessing.py       # Feature engineering
│   ├── models.py               # Model selection
│   └── evaluation.py           # Model evaluation & metrics
│
├── artifacts/
│   ├── trained_model.pkl       # Saved trained model
│   ├── preprocessor.pkl        # Saved preprocessing pipeline
│   └── technical_summary_report.txt
│
├── config/
│   └── config.yaml             # Central configuration file
│
├── main.py                     # Pipeline orchestration script
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation

⚙️ Pipeline Workflow
1️⃣ Data Ingestion

Loads demand, plant, and cost datasets

Merges them into a unified dataset

2️⃣ Data Validation

Verifies required columns

Ensures data consistency

Stops execution if validation fails

3️⃣ Data Preprocessing

Splits features (X) and target (y)

Applies transformations using scikit-learn pipelines

Saves the preprocessor for reuse

4️⃣ Model Training

Trains regression models

Uses cross-validation for robustness

Compares results against a baseline model

5️⃣ Model Evaluation

Evaluates model performance using MSE

Selects the best-performing model

6️⃣ Artifact Generation

Stores trained model

Stores preprocessing pipeline

Generates a technical summary report

🧪 Technologies Used

Python

Pandas / NumPy

Scikit-learn

Joblib

YAML

🚀 How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/your-username/NEC_ML_Pipeline_Group1.git
cd NEC_ML_Pipeline_Group1

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Pipeline
python main.py

4️⃣ Output

Trained model and preprocessing objects saved in artifacts/

Technical evaluation report generated automatically

📊 Outputs & Artifacts

trained_model.pkl – Final regression model

preprocessor.pkl – Feature preprocessing pipeline

technical_summary_report.txt – Model performance summary

✅ Key Features

Modular and scalable design

Config-driven pipeline

Built-in data validation

Baseline model comparison

Reproducible ML workflow

Industry-aligned ML practices

👥 Contributors

Group 1 – NEC ML Pipeline Project

Project developed as a collaborative machine learning assignment focusing on real-world energy optimization challenges.

📌 Future Improvements

Add hyperparameter tuning

Introduce additional regression models

Integrate visualization dashboards

Deploy as a REST API

Extend to real-time demand forecasting

📜 License

This project is developed for educational and research purposes.
