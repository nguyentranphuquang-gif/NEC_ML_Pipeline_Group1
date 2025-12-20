# Import library
import sys

def validate_data(demands, plants, costs):
    # Define required columns
    required_demand_cols = ['Demand ID', 'DF_region', 'DF_daytype'] 
    required_plant_cols = ['Plant ID', 'Plant Type', 'Region']      
    required_cost_cols = ['Demand ID', 'Plant ID', 'Cost_USD_per_MWh']

    def check_cols(df, name, required_cols):
        # Identify missing columns
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

    # Final decision
    if is_valid:
        print("Validation passed. Data is valid")
        return True
    else:
        print("Data structure invalid. Pipeline stopped")
        sys.exit()