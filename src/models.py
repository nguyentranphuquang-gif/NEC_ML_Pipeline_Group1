import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

def get_model(config, tune=False):
    # Retrieve active model name
    model_name = config['model']['active_model']
    seed = config['project']['random_seed']
    
    # Retrieve specific config for that model
    model_config = config['model'][model_name]
    
    # Initialize Base Model
    if model_name == 'random_forest':
        base_model = RandomForestRegressor(random_state=seed)
    elif model_name == 'gradient_boosting':
        base_model = GradientBoostingRegressor(random_state=seed)
    else:
        raise ValueError(f"Error: Model '{model_name}' is not supported")

    # Fixed Parameters (Normal Training)
    if not tune:
        # Get fixed params from config
        params = model_config.get('params', {})
        base_model.set_params(**params)
        
        return base_model

    # Hyperparameter Tuning (Grid Search)
    else:    
        # Get search ranges from config
        param_grid = model_config.get('search_ranges', {})
        
        # Configure GridSearchCV
        search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=3, 
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        return search