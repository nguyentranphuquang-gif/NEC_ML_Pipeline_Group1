# Import library
import pandas as pd

def validate_data(demands, plants, costs, config=None):
    # Required columns from config 
    if config:
        required_demand_cols = config.get('data', {}).get(
            'required_demand_cols',
            ['Demand ID', 'DF_region', 'DF_daytype'] + [f'DF{i}' for i in range(1, 13)]
        )
        required_plant_cols = config.get('data', {}).get(
            'required_plant_cols',
            ['Plant ID', 'Plant Type', 'Region'] + [f'PF{i}' for i in range(1, 19)]
        )
        required_cost_cols = config.get('data', {}).get(
            'required_cost_cols',
            ['Demand ID', 'Plant ID', 'Cost_USD_per_MWh']
        )
        target_col = config.get('data', {}).get('target_col', 'Cost_USD_per_MWh')
    else:
        required_demand_cols = ['Demand ID', 'DF_region', 'DF_daytype'] + [f'DF{i}' for i in range(1, 13)]
        required_plant_cols = ['Plant ID', 'Plant Type', 'Region'] + [f'PF{i}' for i in range(1, 19)]
        required_cost_cols = ['Demand ID', 'Plant ID', 'Cost_USD_per_MWh']
        target_col = 'Cost_USD_per_MWh'

    def check_cols(df, name, required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"[VALIDATION] '{name}' dataset is missing columns: {missing}")

    # Column checks
    check_cols(demands, "Demand", required_demand_cols)
    check_cols(plants, "Plants", required_plant_cols)
    check_cols(costs, "Costs", required_cost_cols)

    # Basic missing summary 
    missing_d = int(demands.isnull().sum().sum())
    if missing_d > 0:
        print(f"[WARN] {missing_d} missing values in demands")

    # Uniqueness checks 
    if demands['Demand ID'].nunique() != len(demands):
        raise ValueError("[VALIDATION] Duplicate Demand IDs found in demands.csv")

    if plants['Plant ID'].nunique() != len(plants):
        raise ValueError("[VALIDATION] Duplicate Plant IDs found in plants.csv")

    # Duplicate (Demand ID, Plant ID) pairs in costs
    if costs.duplicated(subset=['Demand ID', 'Plant ID']).any():
        raise ValueError("[VALIDATION] Duplicate (Demand ID, Plant ID) pairs found in generation_costs.csv")

    # Target missing check (WARN only; policy drop later)
    # Robust: convert to numeric temporarily to detect blanks/strings
    y_tmp = pd.to_numeric(costs[target_col], errors="coerce")
    n_missing_y = int(y_tmp.isna().sum())
    if n_missing_y > 0:
        print(f"[WARN] Missing target '{target_col}': {n_missing_y} rows (should be dropped before training/eval).")

    # Coverage checks (WARN only) 
    n_demands = demands['Demand ID'].nunique()
    n_plants = plants['Plant ID'].nunique()
    expected = n_demands * n_plants
    actual = len(costs)

    if actual != expected:
        print(f"[WARN] costs rows = {actual}, expected = {expected} (= n_demands*n_plants). Check completeness.")

    # Each Demand ID should have n_plants unique plants
    per_demand = costs.groupby('Demand ID')['Plant ID'].nunique()
    if (per_demand < n_plants).any():
        few = per_demand[per_demand < n_plants].head(5)
        print("[WARN] Some Demand IDs have fewer plant entries than expected (example):")
        print(few)

    print("[VALIDATION] Passed structural checks.")
    return True