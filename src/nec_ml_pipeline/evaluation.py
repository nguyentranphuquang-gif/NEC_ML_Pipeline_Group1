# Import libraries
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def selection_metrics(y_true, y_pred, meta, group_col="Demand ID", plant_col="Plant ID"):
    # Ensure alignment
    df = meta[[group_col, plant_col]].reset_index(drop=True).copy()
    df["Actual_Cost"] = pd.Series(y_true).reset_index(drop=True)
    df["Predicted_Cost"] = pd.Series(y_pred).reset_index(drop=True)

    rows = []

    for demand_id, g in df.groupby(group_col, sort=False):
        # chosen plant = min predicted cost
        idx_chosen = g["Predicted_Cost"].idxmin()
        chosen_plant = df.loc[idx_chosen, plant_col]
        true_cost_chosen = float(df.loc[idx_chosen, "Actual_Cost"])

        # true best plant = min actual cost
        idx_best = g["Actual_Cost"].idxmin()
        best_plant = df.loc[idx_best, plant_col]
        true_cost_best = float(df.loc[idx_best, "Actual_Cost"])

        sel_error = true_cost_chosen - true_cost_best  

        rows.append({
            group_col: demand_id,
            "chosen_plant_id": chosen_plant,
            "true_best_plant_id": best_plant,
            "true_cost_chosen": true_cost_chosen,
            "true_cost_best": true_cost_best,
            "selection_error": float(sel_error),
        })

    selection_table = pd.DataFrame(rows)

    mean_sel = float(selection_table["selection_error"].mean()) if len(selection_table) else 0.0
    rmse_sel = float(np.sqrt(np.mean(selection_table["selection_error"] ** 2))) if len(selection_table) else 0.0

    return selection_table, mean_sel, rmse_sel


def evaluate_model(y_true, y_pred, groups, meta=None, dataset_name="Test Set"):
    """
    Returns:
      metrics dict + selection_table (if meta provided)
    """
    # regression RMSE 
    rmse_reg = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    metrics = {
        "rmse_regression": rmse_reg
    }

    selection_table = None
    if meta is not None:
        selection_table, mean_sel, rmse_sel = selection_metrics(
            y_true=y_true,
            y_pred=y_pred,
            meta=meta,
            group_col="Demand ID",
            plant_col="Plant ID",
        )
        metrics["mean_selection_error"] = mean_sel
        metrics["rmse_selection_error"] = rmse_sel

    print(f"[{dataset_name}] RMSE (regression): {rmse_reg:.2f}")
    if meta is not None:
        print(f"[{dataset_name}] Mean selection error: {metrics['mean_selection_error']:.2f}")
        print(f"[{dataset_name}] RMSE selection error: {metrics['rmse_selection_error']:.2f}")

    return metrics, selection_table