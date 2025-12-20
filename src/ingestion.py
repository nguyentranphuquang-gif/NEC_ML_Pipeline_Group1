# Import libraries 
import pandas as pd

def load_data(config):
    
    # Load raw dataset
    demands = pd.read_csv(config['data']['demand_path'], keep_default_na=False, na_values=[''])
    plants = pd.read_csv(config['data']['plant_path'], keep_default_na=False, na_values=[''])
    costs = pd.read_csv(config['data']['cost_path'], keep_default_na=False, na_values=[''])

    return demands, plants, costs

def merge_data(demands, plants, costs):

    # Merge cost and plant dataset (by Plant ID)
    df_tam = pd.merge(costs, plants, on='Plant ID', how='left')
    
    # Merge with demand dataset (by Demand ID)
    master_df = pd.merge(df_tam, demands, on='Demand ID', how='left')

    return master_df