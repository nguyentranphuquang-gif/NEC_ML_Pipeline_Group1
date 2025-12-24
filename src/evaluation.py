import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def calculate_selection_error(y_true, y_pred, groups):
    # Create a temporary DataFrame for calculation
    # Reset_index to ensure alignment if indices are shuffled
    results = pd.DataFrame({
        'Demand ID': groups.values,
        'Actual_Cost': y_true.values,
        'Predicted_Cost': y_pred
    })

    total_error = 0
    valid_groups = 0

    # Iterate through each Demand ID
    for demand_id, group_data in results.groupby('Demand ID'):
        if group_data.empty:
            continue
            
        # Find the index of the plant with the minimum Cost
        idx_selected = group_data['Predicted_Cost'].idxmin()
        
        # Identify the actual cost of the plant the model chose
        cost_of_selected_plant = group_data.loc[idx_selected, 'Actual_Cost']

        # Find the minimum actual cost available in that group
        min_actual_cost = group_data['Actual_Cost'].min()

        # Calculate Error
        error = cost_of_selected_plant - min_actual_cost
        
        total_error += error
        valid_groups += 1

        # Calculate mean error across all Demand ID
    if valid_groups > 0:
        mean_selection_error = total_error / valid_groups
    else:
        mean_selection_error = 0.0

    return mean_selection_error

def evaluate_model(y_true, y_pred, groups, dataset_name="Test Set"):
    # Calculate RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Calculate Selection Error (The NEC KPI)
    selection_error = calculate_selection_error(y_true, y_pred, groups)

    # Print result
    print(f"RMSE: {rmse:.2f}")
    print(f"Selection error: {selection_error:.4f}")
    
    return selection_error, rmse