# Import libraries 
import pandas as pd

def load_data(config):
    try:
        # Load raw dataset with error handling
        demands = pd.read_csv(config['data']['demand_path'], keep_default_na=False, na_values=[''])
        plants = pd.read_csv(config['data']['plant_path'], keep_default_na=False, na_values=[''])
        costs = pd.read_csv(config['data']['cost_path'], keep_default_na=False, na_values=[''])
        
        print(f"Loaded demands: {demands.shape}, plants: {plants.shape}, costs: {costs.shape}")
        
        return demands, plants, costs
    except FileNotFoundError as e:
        raise ValueError(f"File not found: {e}")
    except pd.errors.ParserError as e:
        raise ValueError(f"CSV parse error: {e}")

def merge_data(demands, plants, costs):
    # Merge cost and plant dataset (by Plant ID)
    df_tam = pd.merge(costs, plants, on='Plant ID', how='left')
    
    # Merge with demand dataset (by Demand ID)
    master_df = pd.merge(df_tam, demands, on='Demand ID', how='left')

    return master_df