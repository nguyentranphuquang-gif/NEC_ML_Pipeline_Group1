from __future__ import annotations
from pathlib import Path
from datetime import datetime
import shutil

def make_run_dir(artifacts_root: str | Path, run_name: str | None = None) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    folder = f"run_{run_name}_{ts}" if run_name else f"run_{ts}"
    run_dir = Path(artifacts_root) / folder
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir

def snapshot_config(config_path: str | Path, run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    shutil.copy2(config_path, run_dir / "config_used.yaml")
