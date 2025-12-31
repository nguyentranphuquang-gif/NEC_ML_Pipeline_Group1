import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

def get_model(config, tune=False):
    print("\n==MODEL INITIALIZATION STARTED =")

    # Retrieve active model name
    model_name = config['model']['active_model']
    seed = config['project']['random_seed']

    print(f"[INFO] Active model selected : {model_name}")
    print(f"[INFO] Random seed          : {seed}")
    print(f"[INFO] Hyperparameter tuning: {tune}")

    # Retrieve specific config for that model
    model_config = config['model'][model_name]
    print(f"[INFO] Model config loaded for '{model_name}'")

    # Initialize Base Model
    if model_name == 'random_forest':
        print("[INFO] Initializing RandomForestRegressor")
        base_model = RandomForestRegressor(random_state=seed)

    elif model_name == 'gradient_boosting':
        print("[INFO] Initializing GradientBoostingRegressor")
        base_model = GradientBoostingRegressor(random_state=seed)

    else:
        raise ValueError(f"Error: Model '{model_name}' is not supported")

    # Fixed Parameters (Normal Training)
    if not tune:
        print("[INFO] Running in NORMAL training mode")

        # Get fixed params from config
        params = model_config.get('params', {})
        print(f"[INFO] Applying fixed parameters: {params}")

        base_model.set_params(**params)

        print("[SUCCESS] Model initialization completed")

        return base_model

    # Hyperparameter Tuning (Grid Search)
    else:
        print("[INFO] Running in HYPERPARAMETER TUNING mode")

        # Get search ranges from config
        param_grid = model_config.get('search_ranges', {})
        print(f"[INFO] Grid search parameter ranges: {param_grid}")

        # Configure GridSearchCV
        search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=3,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )

        print("[SUCCESS] GridSearchCV configured successfully")
        return search
