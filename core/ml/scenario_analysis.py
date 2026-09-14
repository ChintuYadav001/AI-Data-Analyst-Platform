"""
core.ml.scenario_analysis
=========================
What-If Scenario Simulator & Sensitivity Analysis for Quantum Analyst ML.
Calculates marginal impact of feature perturbations, outcome elasticity,
and compares baseline predictions against simulated counterfactuals.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from core.ml.prediction import predict_single_record


def simulate_what_if_scenario(
    pipeline: Any,
    base_record: Dict[str, Any],
    modifications: Dict[str, Any],
    task_type: str = "regression",
) -> Dict[str, Any]:
    """
    Simulates a counterfactual What-If scenario against a trained pipeline:
    - Runs base prediction on base_record
    - Applies modifications to generate scenario_record
    - Runs scenario prediction
    - Calculates absolute delta, percentage impact, and feature elasticities
    """
    is_classification = "classification" in task_type

    # 1. Base Prediction
    base_pred_res = predict_single_record(pipeline, base_record, task_type=task_type)
    if "error" in base_pred_res:
        return {"error": f"Base prediction failed: {base_pred_res['error']}"}

    # 2. Construct Scenario Record
    scenario_record = dict(base_record)
    scenario_record.update(modifications)

    # 3. Scenario Prediction
    scenario_pred_res = predict_single_record(pipeline, scenario_record, task_type=task_type)
    if "error" in scenario_pred_res:
        return {"error": f"Scenario prediction failed: {scenario_pred_res['error']}"}

    # 4. Deltas and Elasticity Analysis
    feature_impacts = []
    base_val = base_pred_res.get("prediction")
    scen_val = scenario_pred_res.get("prediction")

    if not is_classification and isinstance(base_val, (int, float)) and isinstance(scen_val, (int, float)):
        delta_y = scen_val - base_val
        pct_change_y = (delta_y / max(1e-5, abs(base_val))) * 100.0

        for feat, new_val in modifications.items():
            orig_val = base_record.get(feat)
            if isinstance(orig_val, (int, float)) and isinstance(new_val, (int, float)):
                delta_x = new_val - orig_val
                pct_change_x = (delta_x / max(1e-5, abs(orig_val))) * 100.0
                elasticity = (pct_change_y / max(1e-5, abs(pct_change_x))) if abs(pct_change_x) > 1e-4 else 0.0
                feature_impacts.append({
                    "feature": feat,
                    "original_value": round(float(orig_val), 4),
                    "scenario_value": round(float(new_val), 4),
                    "delta_x": round(float(delta_x), 4),
                    "pct_change_x": round(float(pct_change_x), 2),
                    "elasticity": round(float(elasticity), 3),
                })
            else:
                feature_impacts.append({
                    "feature": feat,
                    "original_value": str(orig_val),
                    "scenario_value": str(new_val),
                    "delta_x": "Categorical Shift",
                    "pct_change_x": 0.0,
                    "elasticity": 0.0,
                })

        summary = (
            f"Scenario prediction is {scen_val:.2f} vs baseline of {base_val:.2f} "
            f"(Shift: {delta_y:+.2f}, {pct_change_y:+.1f}%). "
            f"Modified {len(modifications)} feature(s)."
        )

        comparison = {
            "task_type": "regression",
            "base_prediction": round(float(base_val), 4),
            "scenario_prediction": round(float(scen_val), 4),
            "delta": round(float(delta_y), 4),
            "pct_change": round(float(pct_change_y), 2),
            "summary": summary,
        }

    else:
        # Classification comparison
        base_class = str(base_val)
        scen_class = str(scen_val)
        base_probs = base_pred_res.get("probabilities", {})
        scen_probs = scenario_pred_res.get("probabilities", {})

        prob_shifts = {}
        for cls_name, p_base in base_probs.items():
            p_scen = scen_probs.get(cls_name, 0.0)
            prob_shifts[cls_name] = round(float(p_scen - p_base), 4)

        for feat, new_val in modifications.items():
            orig_val = base_record.get(feat)
            feature_impacts.append({
                "feature": feat,
                "original_value": str(orig_val),
                "scenario_value": str(new_val),
            })

        class_changed = base_class != scen_class
        summary = (
            f"Classification shifted from '{base_class}' to '{scen_class}'. "
            if class_changed
            else f"Classification remained '{base_class}' with probability shifts."
        )

        comparison = {
            "task_type": "classification",
            "base_prediction": base_class,
            "scenario_prediction": scen_class,
            "class_changed": class_changed,
            "base_probabilities": base_probs,
            "scenario_probabilities": scen_probs,
            "probability_shifts": prob_shifts,
            "summary": summary,
        }

    return {
        "success": True,
        "comparison": comparison,
        "feature_impacts": feature_impacts,
        "base_record": base_record,
        "scenario_record": scenario_record,
        "disclaimer": "Scenario analysis outputs are statistical model estimates and do not guarantee real-world outcomes.",
    }
