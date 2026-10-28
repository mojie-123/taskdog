"""Evaluation metrics aligned with directed axes and base-pose selection."""

from __future__ import annotations

import math

import numpy as np


def axis_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 2)
    prediction /= np.maximum(np.linalg.norm(prediction, axis=1, keepdims=True), 1e-12)
    target /= np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-12)
    cosine = np.clip(np.sum(prediction * target, axis=1), -1.0, 1.0)
    error_deg = np.degrees(np.arccos(cosine))
    return {
        "angular_mean_deg": float(error_deg.mean()),
        "angular_median_deg": float(np.median(error_deg)),
        "within_5_deg": float(np.mean(error_deg <= 5.0)),
        "within_10_deg": float(np.mean(error_deg <= 10.0)),
        "within_15_deg": float(np.mean(error_deg <= 15.0)),
        "flip_rate_over_90_deg": float(np.mean(error_deg > 90.0)),
        "sample_count": int(len(error_deg)),
    }


def _safe_pearson(first: np.ndarray, second: np.ndarray) -> float | None:
    if len(first) < 2 or np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def affordance_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    support_mask: np.ndarray,
    query_xyz_m: np.ndarray,
    *,
    top_k: int = 5,
) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(support_mask, dtype=bool)
    query = np.asarray(query_xyz_m, dtype=np.float64)
    valid_pred = prediction[mask]
    valid_target = target[mask]
    if valid_pred.size == 0:
        raise ValueError("affordance metrics received no supported query points")

    pearsons, similarities = [], []
    selected_values, oracle_values, regrets, distances, topk_hits = [], [], [], [], []
    for sample_index in range(prediction.shape[0]):
        valid = mask[sample_index]
        pred_row = prediction[sample_index, valid]
        target_row = target[sample_index, valid]
        query_row = query[sample_index, valid]
        if len(pred_row) == 0:
            continue
        pearson = _safe_pearson(pred_row, target_row)
        if pearson is not None and math.isfinite(pearson):
            pearsons.append(pearson)
        pred_distribution = pred_row / max(float(pred_row.sum()), 1e-12)
        target_distribution = target_row / max(float(target_row.sum()), 1e-12)
        similarities.append(float(np.minimum(pred_distribution, target_distribution).sum()))

        selected = int(np.argmax(pred_row))
        oracle = int(np.argmax(target_row))
        selected_value = float(target_row[selected])
        oracle_value = float(target_row[oracle])
        selected_values.append(selected_value)
        oracle_values.append(oracle_value)
        regrets.append(oracle_value - selected_value)
        distances.append(float(np.linalg.norm(query_row[selected, :2] - query_row[oracle, :2])))
        k = min(int(top_k), len(pred_row))
        predicted_top = np.argpartition(pred_row, -k)[-k:]
        topk_hits.append(float(oracle in predicted_top))

    return {
        "rmse": float(np.sqrt(np.mean((valid_pred - valid_target) ** 2))),
        "mae": float(np.mean(np.abs(valid_pred - valid_target))),
        "sim": float(np.mean(similarities)),
        "pearson": float(np.mean(pearsons)) if pearsons else 0.0,
        "selected_gt_affordance": float(np.mean(selected_values)),
        "oracle_gt_affordance": float(np.mean(oracle_values)),
        "selection_regret": float(np.mean(regrets)),
        "argmax_distance_m": float(np.mean(distances)),
        f"oracle_in_top_{int(top_k)}": float(np.mean(topk_hits)),
        "sample_count": int(prediction.shape[0]),
    }
