# Import libraries
# main.py
import sys
from pathlib import Path
import json
import yaml
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor

# Ensure imports work (package in ./src)
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import custom modules (use package style)
from src.nec_ml_pipeline.ingestion import load_data, merge_data
from src.nec_ml_pipeline.validation import validate_data
from src.nec_ml_pipeline.preprocessing import prepare_xy, get_preprocessor
from src.nec_ml_pipeline.models import get_model
from src.nec_ml_pipeline.evaluation import evaluate_model
from src.nec_ml_pipeline.utils import make_run_dir, snapshot_config


def _load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_run_dir(config: dict, config_path: str) -> Path:
    out_cfg = config.get("output", {})
    artifacts_root = out_cfg.get("artifacts_root", out_cfg.get("output_dir", "artifacts"))
    run_name = out_cfg.get("run_name", None)
    run_dir = make_run_dir(artifacts_root, run_name=run_name)
    # allow custom snapshot filename if present
    snap_name = out_cfg.get("config_snapshot_filename", "config_used.yaml")
    snapshot_config(config_path, run_dir, filename=snap_name) if "filename" in snapshot_config.__code__.co_varnames else snapshot_config(config_path, run_dir)
    return run_dir


def _write_report(run_dir: Path, lines: list[str], config: dict) -> Path:
    out_cfg = config.get("output", {})
    report_name = out_cfg.get("technical_report_filename", "technical_summary_report.txt")
    report_path = run_dir / report_name
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def group_train_test_split(groups: pd.Series, test_size: float, seed: int):
    """
    Split by unique group IDs (Demand ID). Returns boolean masks for rows.
    """
    uniq = groups.dropna().unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)

    n_test = int(round(len(uniq) * test_size))
    test_groups = set(uniq[:n_test])

    is_test = groups.isin(test_groups).values
    is_train = ~is_test
    return is_train, is_test, sorted(test_groups)


def main():
    config_path = "config/config.yaml"
    report_log = []

    def log(msg: str):
        print(msg)
        report_log.append(msg)

    # [1] Load config + create run dir
    log("Loading Configuration...")
    try:
        config = _load_config(config_path)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    run_dir = _resolve_run_dir(config, config_path)
    log(f"[INFO] Run directory: {run_dir}")

    # [2] Ingestion + validation + merge
    log("\n[2] Ingestion & Validation")
    demands, plants, costs = load_data(config)
    validate_data(demands, plants, costs, config)

    master_df = merge_data(demands, plants, costs)
    log(f"Master data shape: {master_df.shape}")

    # [3] Prepare X/y/groups/meta (meta includes Demand ID + Plant ID)
    log("\n[3] Prepare X/y/groups/meta")
    X, y, groups, meta = prepare_xy(master_df, config)
    log(f"X: {X.shape} | y: {len(y)} | groups: {len(groups)} | meta: {meta.shape}")

    # [4] Grouped held-out split (by Demand ID)
    eval_cfg = config.get("evaluation", {})
    test_size = float(eval_cfg.get("test_size", 0.2))
    seed = int(config["project"]["random_seed"])

    is_train, is_test, test_group_ids = group_train_test_split(groups, test_size=test_size, seed=seed)

    X_train, X_test = X.loc[is_train], X.loc[is_test]
    y_train, y_test = y.loc[is_train], y.loc[is_test]
    groups_train, groups_test = groups.loc[is_train], groups.loc[is_test]
    meta_train, meta_test = meta.loc[is_train], meta.loc[is_test]

    log(f"\n[4] Held-out split by Demand ID")
    log(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    log(f"Unique train demands: {groups_train.nunique()} | Unique test demands: {groups_test.nunique()}")

    # [5] Baseline (Dummy) evaluated on held-out
    log("\n[5] Baseline (DummyRegressor) on held-out")
    baseline_pipe = Pipeline(steps=[
        ("preprocess", get_preprocessor(X_train, config=config)),
        ("model", DummyRegressor(strategy="mean"))
    ])
    baseline_pipe.fit(X_train, y_train)
    y_pred_base = baseline_pipe.predict(X_test)

    base_metrics, base_sel_table = evaluate_model(
        y_true=y_test,
        y_pred=y_pred_base,
        groups=groups_test,
        meta=meta_test,
        dataset_name="Held-out | BASELINE"
    )
    log(f"Baseline metrics: {base_metrics}")

    # [6] Untuned model trained on train, evaluated on held-out
    log(f"\n[6] Untuned model on held-out: {config['model']['active_model']}")
    untuned_model = get_model(config, tune=False)
    untuned_pipe = Pipeline(steps=[
        ("preprocess", get_preprocessor(X_train, config=config)),
        ("model", untuned_model)
    ])
    untuned_pipe.fit(X_train, y_train)
    y_pred_untuned = untuned_pipe.predict(X_test)

    untuned_metrics, untuned_sel_table = evaluate_model(
        y_true=y_test,
        y_pred=y_pred_untuned,
        groups=groups_test,
        meta=meta_test,
        dataset_name="Held-out | UNTUNED"
    )
    log(f"Untuned metrics: {untuned_metrics}")

    # [7] Hyperparameter tuning (GridSearchCV over Pipeline) on TRAIN only
    log("\n[7] Hyperparameter tuning (GridSearchCV) on TRAIN only")
    searcher = get_model(config, X=X_train, meta=meta_train, tune=True)
    searcher.fit(X_train, y_train, groups=groups_train)

    best_pipeline = searcher.best_estimator_
    best_params = searcher.best_params_
    best_score = float(searcher.best_score_)  # negative rmse_selection_error
    best_cv_rmse_sel = float(-best_score)

    log("TUNING RESULTS:")
    log(f"Best params: {best_params}")
    log(f"Best CV RMSE selection error (converted): {best_cv_rmse_sel:.4f}")

    # Evaluate tuned on held-out
    log("\n[8] Tuned model evaluation on held-out")
    y_pred_tuned = best_pipeline.predict(X_test)

    tuned_metrics, tuned_sel_table = evaluate_model(
        y_true=y_test,
        y_pred=y_pred_tuned,
        groups=groups_test,
        meta=meta_test,
        dataset_name="Held-out | TUNED"
    )
    log(f"Tuned metrics: {tuned_metrics}")

    # [9] Save artefacts to run_dir
    log("\n[9] Saving artefacts")

    out_cfg = config.get("output", {})
    pipeline_name = out_cfg.get("pipeline_filename", "model_pipeline.joblib")
    joblib.dump(best_pipeline, run_dir / pipeline_name)
    log(f"Saved final pipeline: {run_dir / pipeline_name}")

    # cv results
    if hasattr(searcher, "cv_results_"):
        cv_name = out_cfg.get("cv_results_filename", "cv_results.csv")
        pd.DataFrame(searcher.cv_results_).to_csv(run_dir / cv_name, index=False)
        log(f"Saved CV results: {run_dir / cv_name}")

    # selection tables
    sel_name = out_cfg.get("selection_table_filename", out_cfg.get("selection_table_path", "selection_table.csv"))
    # save tuned selection table as main
    if tuned_sel_table is not None:
        tuned_sel_table.to_csv(run_dir / sel_name, index=False)
        log(f"Saved tuned selection table: {run_dir / sel_name}")

    # also save untuned selection table for comparison
    if untuned_sel_table is not None:
        untuned_path = run_dir / "selection_table_untuned.csv"
        untuned_sel_table.to_csv(untuned_path, index=False)
        log(f"Saved untuned selection table: {untuned_path}")

    # held-out summary json
    heldout_name = out_cfg.get("heldout_results_filename", "heldout_results.json")
    payload = {
        "active_model": config["model"]["active_model"],
        "seed": seed,
        "test_size_demands": test_size,
        "n_test_demands": int(groups_test.nunique()),
        "n_train_demands": int(groups_train.nunique()),
        "best_params": best_params,
        "best_cv_rmse_selection_error": best_cv_rmse_sel,
        "baseline_metrics": base_metrics,
        "untuned_metrics": untuned_metrics,
        "tuned_metrics": tuned_metrics,
    }
    (run_dir / heldout_name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"Saved heldout summary: {run_dir / heldout_name}")

    # report
    report_path = _write_report(run_dir, report_log, config)
    log(f"\nTechnical Summary Report: {report_path}")
    log("\nPIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
