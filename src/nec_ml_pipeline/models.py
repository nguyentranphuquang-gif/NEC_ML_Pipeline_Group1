# Import libraries
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, GroupKFold
from sklearn.pipeline import Pipeline

from nec_ml_pipeline.preprocessing import get_preprocessor
from nec_ml_pipeline.evaluation import selection_metrics  


def _prefix_param_grid(param_grid: dict, prefix: str = "model__") -> dict:
    new_grid = {}
    for k, v in (param_grid or {}).items():
        if k.startswith(prefix):
            new_grid[k] = v
        else:
            new_grid[f"{prefix}{k}"] = v
    return new_grid


def _make_selection_rmse_scorer(meta: pd.DataFrame,
                                group_col: str = "Demand ID",
                                plant_col: str = "Plant ID"):
    """
    Returns a callable scorer(estimator, X, y) -> score.
    """
    def scorer(estimator, X, y_true):
        # X is a DataFrame subset with original index preserved
        y_pred = estimator.predict(X)
        y_true = pd.Series(y_true, index=X.index)

        # align meta to X rows
        meta_sub = meta.loc[X.index, [group_col, plant_col]]

        # compute selection RMSE
        _, _, rmse_sel = selection_metrics(
            y_true=y_true,
            y_pred=y_pred,
            meta=meta_sub,
            group_col=group_col,
            plant_col=plant_col
        )
        return -rmse_sel

    return scorer


def _build_base_model(config, model_name: str):
    seed = config["project"]["random_seed"]

    if model_name == "random_forest":
        return RandomForestRegressor(random_state=seed)
    elif model_name == "gradient_boosting":
        return GradientBoostingRegressor(random_state=seed)
    else:
        raise ValueError(f"Model '{model_name}' is not supported")


def get_model(config, X=None, meta=None, tune: bool = False):
    """
    - tune=False: return a configured regressor (untuned, fixed params from config)
    - tune=True : return GridSearchCV over a Pipeline(preprocess+model) using LOGO + selection RMSE scorer
    """
    print("\n MODEL INITIALIZATION STARTED")

    model_name = config["model"]["active_model"]
    model_config = config["model"][model_name]

    print(f"[INFO] Active model selected : {model_name}")
    print(f"[INFO] Hyperparameter tuning: {tune}")

    base_model = _build_base_model(config, model_name)

    # Normal training mode: fixed params
    if not tune:
        params = model_config.get("params", {})
        print(f"[INFO] Applying fixed parameters: {params}")
        base_model.set_params(**params)
        print("[SUCCESS] Model initialization completed (untuned)")
        return base_model

    # Tuning mode: GridSearchCV over Pipeline with LOGO + selection scorer
    if X is None or meta is None:
        raise ValueError("When tune=True, you must provide X (DataFrame) and meta (DataFrame).")

    # Build preprocess + model pipeline
    preprocessor = get_preprocessor(X, config=config)
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", base_model)
    ])

    # Parameter grid (auto prefix with model__)
    raw_grid = model_config.get("search_ranges", {})
    param_grid = _prefix_param_grid(raw_grid, prefix="model__")
    print(f"[INFO] Grid search parameter ranges (pipeline): {param_grid}")

    # CV splitter: LOGO if requested, else GroupKFold
    eval_cfg = config.get("evaluation", {})
    cv_mode = eval_cfg.get("cv", "logo")
    if cv_mode == "logo":
        cv = LeaveOneGroupOut()
        print("[INFO] CV splitter: LeaveOneGroupOut (LOGO)")
    else:
        n_splits = int(eval_cfg.get("n_splits", 5))
        cv = GroupKFold(n_splits=n_splits)
        print(f"[INFO] CV splitter: GroupKFold(n_splits={n_splits})")

    scorer = _make_selection_rmse_scorer(meta, group_col="Demand ID", plant_col="Plant ID")

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scorer,
        n_jobs=4,
        refit=True,
        verbose=1,
        return_train_score=True
    )

    print("[SUCCESS] GridSearchCV configured successfully (selection RMSE + LOGO)")
    return search
