# Import libraries
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def prepare_xy(master_df, config):
    target_col = config['data']['target_col']      # 'Cost_USD_per_MWh'
    group_col = config['data']['group_col_split']  # 'Demand ID'

    # Drop rows where 'Cost_USD_per_MWh' is NaN
    master_df = master_df.dropna(subset=[target_col]).copy()

    # Meta for selection-error evaluation 
    meta = master_df[[group_col, 'Plant ID']].copy()

    # Drop columns not used as features
    drop_cols = [target_col, group_col, 'Plant ID']

    X = master_df.drop(columns=drop_cols)
    y = master_df[target_col]
    groups = master_df[group_col]

    return X, y, groups, meta

def get_preprocessor(X, config=None):
    numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    transformers = []
    if numerical_cols:
        transformers.append(("num", num_transformer, numerical_cols))
    if categorical_cols:
        transformers.append(("cat", cat_transformer, categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False
    )
    return preprocessor
    