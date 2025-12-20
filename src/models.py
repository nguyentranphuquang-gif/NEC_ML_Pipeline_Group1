import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def get_model(config):
    # Retrieve the active model name 
    model_name = config['model']['active_model']

    # Retrieve specific hyperparameters
    model_params = config['model'][model_name]

    # Retrieve random_seed for reproducibility
    seed = config['project']['random_seed']

    print(f"Main model: {model_name.upper()}")
    print(f"Hyperparameters: {model_params}")

    # Model Dispatcher
    if model_name == 'random_forest':
        return RandomForestRegressor(random_state=seed, **model_params)
    
    elif model_name == 'gradient_boosting':
        return GradientBoostingRegressor(random_state=seed, **model_params)
    
    else:
        raise ValueError(f"Error: Model '{model_name}' is not supported")