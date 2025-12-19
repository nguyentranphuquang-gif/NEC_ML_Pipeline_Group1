# Import libraries
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_and_merge_data(demand_path, plant_path, cost_path):

    demands = pd.read_csv(demand_path, keep_default_na=False, na_values=[''])
    plants = pd.read_csv(plant_path, keep_default_na=False, na_values=[''])
    costs = pd.read_csv(cost_path, keep_default_na=False, na_values=[''])
    
    merged_df = pd.merge(costs, plants, on='Plant ID', how='left')
    master_df = pd.merge(merged_df, demands, on='Demand ID', how='left')

    master_df = master_df.dropna(subset=['Cost_USD_per_MWh'])
    
    print(f"Master dataset shape: {master_df.shape}")
    return master_df

def prepare_xy(master_df):

    target_col = 'Cost_USD_per_MWh'
    group_col = 'Demand ID'
    drop_cols = [target_col, group_col, 'Plant ID']

    X = master_df.drop(columns=drop_cols)
    y = master_df[target_col]
    groups = master_df[group_col]

    return X, y, groups

def get_preprocessor(X):

    numerical_cols = X.select_dtypes(include=['float64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    # Mean or median
    num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
    
    cat_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numerical_cols),
            ('cat', cat_transformer, categorical_cols)
        ],
        verbose_feature_names_out=False
    )

    return preprocessor

    