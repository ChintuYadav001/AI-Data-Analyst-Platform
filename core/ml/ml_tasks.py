"""
core.ml.ml_tasks
================
Background task execution integrating ML training and benchmarking with JobManager.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from core.job_manager import JobManager
from core.ml.model_selection import run_benchmark
from core.ml.tuning import tune_model_hyperparameters
from core.ml.experiment_tracker import save_experiment


ML_BENCHMARK_STAGES = [
    "STAGE 1: Preprocessing & Splitting",
    "STAGE 2: Model Training & CV",
    "STAGE 3: Metric Evaluation & Diagnostics",
    "STAGE 4: Champion Selection & Registry",
    "STAGE 5: Ready",
]


def start_background_benchmark(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    task_type: str,
    compute_profile: str = "Balanced",
    model_ids: Optional[List[str]] = None,
    dataset_name: str = "dataset",
) -> str:
    """Submits ML benchmarking job to background thread pool."""
    def _worker(job_id: str):
        def progress_cb(pct: float, msg: str):
            stage_idx = min(int(pct * len(ML_BENCHMARK_STAGES)), len(ML_BENCHMARK_STAGES) - 1)
            stage_name = ML_BENCHMARK_STAGES[stage_idx]
            JobManager.update_progress(job_id, int(pct * 90), stage_name, msg)

        res = run_benchmark(
            df=df,
            target_col=target_col,
            feature_cols=feature_cols,
            task_type=task_type,
            compute_profile=compute_profile,
            model_ids=model_ids,
            progress_callback=progress_cb,
        )

        # Save champion model to experiment registry if available
        champ_id = res.get("champion_id")
        if champ_id and champ_id in res["all_results"]:
            champ_res = res["all_results"][champ_id]
            exp_id = save_experiment(
                pipeline=champ_res["pipeline"],
                model_id=champ_id,
                model_name=champ_res["model_name"],
                task_type=task_type,
                target_col=target_col,
                feature_cols=feature_cols,
                metrics=champ_res["metrics"],
                dataset_name=dataset_name,
                is_champion=True,
            )
            res["experiment_id"] = exp_id

        JobManager.update_progress(job_id, 100, "STAGE 5: Ready", "Benchmark finished successfully.")
        return res

    job_id = JobManager.submit(
        job_type="ml_benchmark",
        fn=_worker,
        stages=ML_BENCHMARK_STAGES,
        metadata={"target": target_col, "task": task_type, "profile": compute_profile},
    )
    return job_id
