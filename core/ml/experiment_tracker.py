"""
core.ml.experiment_tracker
==========================
Experiment registry, versioning, serialization (.joblib), and rollback management.
Persists metadata and model pipelines under data/experiments/.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
import joblib

EXPERIMENTS_DIR = os.path.join(os.getcwd(), "data", "experiments")


def ensure_experiments_dir():
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)


def save_experiment(
    pipeline: Any,
    model_id: str,
    model_name: str,
    task_type: str,
    target_col: str,
    feature_cols: List[str],
    metrics: Dict[str, Any],
    dataset_name: str = "default_dataset",
    is_champion: bool = False,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Saves trained pipeline artifact and writes manifest metadata.
    Returns generated experiment_id.
    """
    ensure_experiments_dir()
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    exp_id = f"exp_{timestamp_str}_{model_id}"
    exp_dir = os.path.join(EXPERIMENTS_DIR, exp_id)
    os.makedirs(exp_dir, exist_ok=True)

    # Serialize pipeline
    model_path = os.path.join(exp_dir, "model.joblib")
    joblib.dump(pipeline, model_path)

    # Save metadata JSON
    manifest = {
        "experiment_id": exp_id,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": model_id,
        "model_name": model_name,
        "task_type": task_type,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "dataset_name": dataset_name,
        "is_champion": is_champion,
        "model_file": "model.joblib",
        "extra_meta": extra_meta or {},
    }

    manifest_path = os.path.join(exp_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # If champion, update champion pointer
    if is_champion:
        set_active_champion(exp_id)

    return exp_id


def list_experiments() -> List[Dict[str, Any]]:
    """Lists all stored experiments sorted from newest to oldest."""
    ensure_experiments_dir()
    experiments = []

    if not os.path.exists(EXPERIMENTS_DIR):
        return experiments

    for item in os.listdir(EXPERIMENTS_DIR):
        item_path = os.path.join(EXPERIMENTS_DIR, item)
        manifest_file = os.path.join(item_path, "manifest.json")
        if os.path.isdir(item_path) and os.path.exists(manifest_file):
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    experiments.append(data)
            except Exception:
                pass

    experiments.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return experiments


def get_experiment(exp_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves manifest metadata for an experiment."""
    manifest_path = os.path.join(EXPERIMENTS_DIR, exp_id, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_model(exp_id: str) -> Any:
    """Loads a serialized joblib pipeline for a given experiment ID."""
    model_path = os.path.join(EXPERIMENTS_DIR, exp_id, "model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model artifact found for experiment '{exp_id}'.")
    return joblib.load(model_path)


def set_active_champion(exp_id: str) -> None:
    """Sets an experiment as the active champion model."""
    ensure_experiments_dir()
    champ_pointer = os.path.join(EXPERIMENTS_DIR, "active_champion.json")
    meta = get_experiment(exp_id)
    if meta:
        with open(champ_pointer, "w", encoding="utf-8") as f:
            json.dump({"active_champion_id": exp_id, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)


def get_active_champion() -> Optional[Dict[str, Any]]:
    """Gets the currently active champion experiment metadata."""
    champ_pointer = os.path.join(EXPERIMENTS_DIR, "active_champion.json")
    if os.path.exists(champ_pointer):
        try:
            with open(champ_pointer, "r", encoding="utf-8") as f:
                data = json.load(f)
                exp_id = data.get("active_champion_id")
                return get_experiment(exp_id)
        except Exception:
            pass

    # Fallback to latest experiment
    all_exp = list_experiments()
    return all_exp[0] if all_exp else None
