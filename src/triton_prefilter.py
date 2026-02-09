"""Lightweight pre-filter for Triton autotune num_stages candidates."""

from __future__ import division

from gpu_specs import get_gpu_spec
from predictor import predict_one


def prune_num_stages(
    workload,
    gpu_name,
    problem_size,
    candidate_stages,
    epsilon=0.03,
    tile_size=None,
):
    gpu = get_gpu_spec(gpu_name)

    # Boundary behavior in proposal: V100 without cp.async should return stage 1 only.
    if not gpu["supports_cp_async"]:
        return {
            "kept": [1],
            "pruned": [s for s in candidate_stages if int(s) != 1],
            "scores": [{"stage": 1, "pred_speedup": 1.0, "valid": True}],
            "reason": "gpu_has_no_cp_async",
        }

    scores = []
    for stage in candidate_stages:
        stage = int(stage)
        pred = predict_one(workload, gpu_name, problem_size, stage, tile_size=tile_size)
        scores.append(
            {
                "stage": stage,
                "pred_speedup": float(pred.get("pred_speedup", 0.0)),
                "valid": bool(pred.get("valid", False)),
                "reason": pred.get("reason", ""),
            }
        )

    kept = []
    for item in scores:
        s = int(item["stage"])
        speedup = float(item["pred_speedup"])
        valid = bool(item["valid"])

        if s == 1:
            kept.append(1)
            continue

        if valid and speedup > (1.0 + float(epsilon)):
            kept.append(s)

    if len(kept) == 0:
        # Safety: never drop all candidates.
        best = sorted(scores, key=lambda x: x["pred_speedup"], reverse=True)[0]
        kept = [int(best["stage"])]

    kept = sorted(set(kept))
    pruned = sorted(set(int(x) for x in candidate_stages if int(x) not in kept))

    return {
        "kept": kept,
        "pruned": pruned,
        "scores": scores,
        "reason": "epsilon_prune",
    }
