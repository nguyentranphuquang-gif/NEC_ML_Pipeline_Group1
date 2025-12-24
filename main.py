# Import libraries
import pandas as pd
import numpy as np
import yaml
import joblib
import os
import sys
from sklearn.model_selection import GroupKFold
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error

# Import custom modules 
from src.ingestion import load_data, merge_data
from src.validation import validate_data
from src.preprocessing import prepare_xy, get_preprocessor
from src.models import get_model
from src.evaluation import evaluate_model

def main():
    # [1] CONFIGURATION & SETUP 
    print("Loading Configuration")
    try:
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Create artifacts directory 
    os.makedirs(config['output']['output_dir'], exist_ok=True)
    
    # Initialize report buffer
    report_log = []
    def log(msg): 
        print(msg)
        report_log.append(msg)

    # [2] INGESTION & VALIDATION 
    log("\n Ingestion & Validation")
    demands, plants, costs = load_data(config)
    validate_data(demands, plants, costs, config)
    master_df = merge_data(demands, plants, costs)
    
    log(f"Data shape: {master_df.shape}")

    # [3] PREPROCESSING 
    log("\n Preprocessing")
    X, y, groups = prepare_xy(master_df, config)
    
    # Build and Fit Preprocessor
    preprocessor = get_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)
    
    # Save Preprocessor 
    prep_path = f"{config['output']['output_dir']}/{config['output']['preprocessor_path']}"
    joblib.dump(preprocessor, prep_path)
    log(f"Preprocessor artefact saved to: {prep_path}")

    # [4] BASELINE COMPARISON 
    log("\n Baseline Evaluation")

    # Dummy Regressor predicts the mean value always
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_processed, y)
    y_base = dummy.predict(X_processed)
    base_rmse = np.sqrt(mean_squared_error(y, y_base))
    
    log(f"BASELINE RMSE (Dummy Model): {base_rmse:.2f}")

    # [5] UNTUNED MODEL EVALUATION 
    active_model = config['model']['active_model']
    log(f"\n Evaluating Active Model: {active_model.upper()} (Untuned)")
    
    # Setup LOGO Cross-Validation
    n_splits = config['evaluation']['n_splits']
    gkf = GroupKFold(n_splits=n_splits)
    
    sel_errors = []
    rmse_scores = []

    # Iterate through folds
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_processed, y, groups)):
        # Split Data respecting Groups
        X_train, X_test = X_processed[train_idx], X_processed[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_test = groups.iloc[test_idx]

        # Get Model with Fixed Params (tune=False)
        model = get_model(config, tune=False)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Evaluate using Custom Metric
        sel, rmse = evaluate_model(y_test, y_pred, groups_test, dataset_name=f"Fold {fold+1}")
        sel_errors.append(sel)
        rmse_scores.append(rmse)

    avg_sel = np.mean(sel_errors)
    avg_rmse = np.mean(rmse_scores)
    
    log(f"{active_model.upper()} PERFORMANCE (Untuned):")
    log(f"Average Selection Error: {avg_sel:.2f}")
    log(f"Average RMSE: {avg_rmse:.2f}")

    # [6] HYPER-PARAMETER OPTIMISATION 
    log(f"\n Hyper-parameter Optimisation (GridSearchCV)")
    
    # Get GridSearchCV Object (tune=True)
    searcher = get_model(config, tune=True)
    
    # Run Grid Search
    searcher.fit(X_processed, y)
    
    # Extract Best Results
    best_model = searcher.best_estimator_
    best_params = searcher.best_params_
    best_cv_rmse = -searcher.best_score_ # Convert neg_rmse to positive

    log(f"\n OPTIMIZATION RESULTS:")
    log(f"Best Parameters: {best_params}")
    log(f"Best CV RMSE: {best_cv_rmse:.2f}")
    
    # Calculate improvement
    improvement = base_rmse - best_cv_rmse
    log(f"Improvement over Baseline: {improvement:.2f}")

    # [7] PACKAGING & ARTEFACTS 
    log("\n Packaging & Handover")
    
    # Save Final Model Artefact
    model_path = f"{config['output']['output_dir']}/{config['output']['model_path']}"
    joblib.dump(best_model, model_path)
    log(f"Saved Final Model to: {model_path}")

    # Generate Technical Summary Data 
    report_path = f"{config['output']['output_dir']}/technical_summary_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_log))
    
    print(f"\n PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Summary Report: {report_path}")

if __name__ == "__main__":
    main()