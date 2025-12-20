# Import libraries
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer



def prepare_xy(master_df):
    # Drop rows where 'Cost_USD_per_MWh' is NaN
    master_df = master_df.dropna(subset=['Cost_USD_per_MWh'])

    # Define columns
    target_col = 'Cost_USD_per_MWh'
    group_col = 'Demand ID'
    drop_cols = [target_col, group_col, 'Plant ID']

    # Split X,y,groups
    X = master_df.drop(columns=drop_cols)
    y = master_df[target_col]
    groups = master_df[group_col]

    return X, y, groups

def get_preprocessor(X):

    numerical_cols = X.select_dtypes(include=['float64','int64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    # Impute mean for missing values
    num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
    # Encode categorical columns
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

    