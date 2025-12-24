# Import library
import sys
import pandas as pd

def validate_data(demands, plants, costs, config=None):
    if config:
        required_demand_cols = config.get('data', {}).get('required_demand_cols', ['Demand ID', 'DF_region', 'DF_daytype'] + [f'DF{i}' for i in range(1, 13)])
        required_plant_cols = config.get('data', {}).get('required_plant_cols', ['Plant ID', 'Plant Type', 'Region'] + [f'PF{i}' for i in range(1, 19)])
        required_cost_cols = config.get('data', {}).get('required_cost_cols', ['Demand ID', 'Plant ID', 'Cost_USD_per_MWh'])
    else:
        required_demand_cols = ['Demand ID', 'DF_region', 'DF_daytype'] + [f'DF{i}' for i in range(1, 13)]  # Add full DF
        required_plant_cols = ['Plant ID', 'Plant Type', 'Region'] + [f'PF{i}' for i in range(1, 19)]  # Add full PF
        required_cost_cols = ['Demand ID', 'Plant ID', 'Cost_USD_per_MWh']

    def check_cols(df, name, required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"'{name}' dataset is missing columns: {missing}")
            return False
        return True

    # Run checks for all datasets
    is_valid = True
    if not check_cols(demands, "Demand", required_demand_cols): is_valid = False
    if not check_cols(plants, "Plants", required_plant_cols): is_valid = False
    if not check_cols(costs, "Costs", required_cost_cols): is_valid = False

    # Check for missing values, uniques
    if is_valid:
        # Missing count
        missing_d = demands.isnull().sum().sum()
        if missing_d > 0:
            print(f"{missing_d} missing values in demands")
        
        # Uniques: IDs unique
        if len(plants['Plant ID'].unique()) != len(plants):
            print("Duplicate Plant IDs")
            is_valid = False

    # Final decision
    if is_valid:
        print("Validation passed. Data is valid")
        return True
    else:
        print("Data structure invalid. Pipeline stopped")
        sys.exit()