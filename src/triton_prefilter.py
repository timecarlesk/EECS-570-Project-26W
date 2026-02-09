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
    l2_mode="effective",
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
        pred = predict_one(
            workload,
            gpu_name,
            problem_size,
            stage,
            tile_size=tile_size,
            l2_mode=l2_mode,
        )
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


def _infer_problem_size(named_args, problem_size_keys):
    if callable(problem_size_keys):
        return int(problem_size_keys(named_args))

    if isinstance(problem_size_keys, (list, tuple)):
        vals = []
        for key in problem_size_keys:
            if key in named_args:
                try:
                    vals.append(int(named_args[key]))
                except Exception:
                    pass
        if vals:
            return max(vals)

    for key in ("N", "n", "M", "m", "K", "k", "problem_size"):
        if key in named_args:
            try:
                return int(named_args[key])
            except Exception:
                pass

    return 0


def _infer_tile_size(config, named_args, tile_size_keys):
    if callable(tile_size_keys):
        return tile_size_keys(config, named_args)

    if not isinstance(tile_size_keys, (list, tuple)):
        tile_size_keys = ["BLOCK_SIZE", "BLOCK_M", "BLOCK_SIZE_M", "BLOCK_SIZE_N"]

    cfg_kwargs = getattr(config, "kwargs", {})
    for key in tile_size_keys:
        if key in cfg_kwargs:
            try:
                return int(cfg_kwargs[key])
            except Exception:
                pass

    for key in tile_size_keys:
        if key in named_args:
            try:
                return int(named_args[key])
            except Exception:
                pass

    return None


def make_stage_prune_callback(
    workload,
    gpu_name,
    epsilon=0.03,
    problem_size_keys=("N",),
    tile_size_keys=("BLOCK_SIZE", "BLOCK_M", "BLOCK_SIZE_M", "BLOCK_SIZE_N"),
    l2_mode="effective",
):
    """Return a Triton-compatible early_config_prune callback.

    Callback signature follows Triton: fn(configs, named_args) -> filtered_configs.
    """

    def _callback(configs, named_args):
        if not configs:
            return configs

        problem_size = _infer_problem_size(named_args, problem_size_keys)
        if problem_size <= 0:
            return configs

        stages = []
        for cfg in configs:
            stages.append(int(getattr(cfg, "num_stages", 1)))

        tile_size = _infer_tile_size(configs[0], named_args, tile_size_keys)
        prune = prune_num_stages(
            workload=workload,
            gpu_name=gpu_name,
            problem_size=problem_size,
            candidate_stages=stages,
            epsilon=epsilon,
            tile_size=tile_size,
            l2_mode=l2_mode,
        )

        kept = set(int(s) for s in prune["kept"])
        filtered = [cfg for cfg in configs if int(getattr(cfg, "num_stages", 1)) in kept]
        return filtered if filtered else configs

    return _callback


def autotune_with_cp_async_prefilter(
    configs,
    key,
    workload,
    gpu_name,
    epsilon=0.03,
    problem_size_keys=("N",),
    tile_size_keys=("BLOCK_SIZE", "BLOCK_M", "BLOCK_SIZE_M", "BLOCK_SIZE_N"),
    l2_mode="effective",
    **autotune_kwargs
):
    """Build a Triton autotune decorator with predictor-based early pruning.

    Usage:
      @autotune_with_cp_async_prefilter(
          configs=[...],
          key=['M','N','K'],
          workload='gemm',
          gpu_name='A100_MIG_3g40gb',
      )
      @triton.jit
      def kernel(...):
          ...
    """

    try:
        import triton
    except Exception as e:
        raise RuntimeError("Triton is required for decorator integration: {}".format(e))

    user_prune = autotune_kwargs.pop("prune_configs_by", {})
    if user_prune is None:
        user_prune = {}

    user_early = user_prune.get("early_config_prune", None)

    prefilter_cb = make_stage_prune_callback(
        workload=workload,
        gpu_name=gpu_name,
        epsilon=epsilon,
        problem_size_keys=problem_size_keys,
        tile_size_keys=tile_size_keys,
        l2_mode=l2_mode,
    )

    if user_early is None:
        combined_early = prefilter_cb
    else:
        def combined_early(configs_in, named_args):
            cfgs = user_early(configs_in, named_args)
            return prefilter_cb(cfgs, named_args)

    prune_configs_by = dict(user_prune)
    prune_configs_by["early_config_prune"] = combined_early

    return triton.autotune(
        configs=configs,
        key=key,
        prune_configs_by=prune_configs_by,
        **autotune_kwargs
    )
