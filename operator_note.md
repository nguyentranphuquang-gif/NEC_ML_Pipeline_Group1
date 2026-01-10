# Operator Note — NEC ML Pipeline (Group 1)

## Purpose
This pipeline predicts generation cost (USD/MWh) for each plant under a demand scenario and selects the minimum-cost plant per Demand ID.

## How to run
1. Ensure the three input files are present under `data/`:
   - `demand.csv`, `plants.csv`, `generation_costs.csv`
2. Adjust settings in `config/config.yaml` 
3. Run:
   - `python main.py`

## Outputs (per run)
A run folder is created under `artifacts/run_YYYYMMDD_HHMM/` containing:
- `model_pipeline.joblib`: fitted preprocessing + model pipeline
- `config_used.yaml`: exact configuration snapshot for this run
- `cv_results.csv`: LOGO CV results for hyperparameter search (candidate leaderboard)
- `heldout_results.json`: held-out metrics (baseline, untuned, tuned)
- `selection_table.csv`: per-demand chosen plant vs true best plant and selection error
- `technical_summary_report.txt`: execution log

## How to interpret results
- Use `heldout_results.json` for headline performance (RMSE regression + selection metrics).
- Use `selection_table.csv` to inspect the most costly mis-selections and recurring patterns (region/daytype/plant type).

