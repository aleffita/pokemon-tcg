with open("scratch/stage4_native_mlx/main.py", "r") as f:
    text = f.read()

text = text.replace("import torch\nfrom rl.policy_infer_torch import load_inference_checkpoint", 
"""import mlx.core as mx
import mlx.nn as nn
from rl.policy_mlx import build_token_net_mlx
import pickle""")

text = text.replace("""def _load_model():
    if _MODEL_PATH is None:
        return None, {}, {
            "version": 1, "seed": 0, "bc_would_ko": False, "bc_wk_nvar": 10,
            "provenance": "no-checkpoint",
        }
    net, metadata = load_inference_checkpoint(_MODEL_PATH, _CARD_TABLE)
    runtime_cfg = metadata["inference_config"]
    print(f"[bc-agent] loaded PyTorch FP16 model {_MODEL_PATH} "
          f"(nlayers={metadata['nlayers']}, "
          f"scratch={metadata['scratch_registers']}, "
          f"would_ko={runtime_cfg['bc_would_ko']}, "
          f"mode={_INFERENCE_MODE})")
    return net, metadata, runtime_cfg""", 
"""def _load_model():
    if _MODEL_PATH is None:
        return None, {}, {"version": 1, "seed": 0, "bc_would_ko": False, "bc_wk_nvar": 10, "provenance": "no-checkpoint"}
    with open(_MODEL_PATH, "rb") as f:
        state = pickle.load(f)
    ckpt_cfg = state.get("arch_config", state.get("config", state.get("net_config", {})))
    cfg = {"d_model": 128, "nhead": 4, "nlayers": 4, "static": False, "split_heads": False, "structured": True, "scratch_registers": 32, "value_atoms": 0, "value_vmax": 0.0}
    for key in ("d_model", "nhead", "nlayers", "ff_dim", "static", "split_heads", "structured", "scratch_registers", "value_atoms", "value_vmax"):
        if key in ckpt_cfg: cfg[key] = ckpt_cfg[key]
    if "ff_dim" in ckpt_cfg: cfg["ff"] = ckpt_cfg["ff_dim"]
    net = build_token_net_mlx(_CARD_TABLE, cfg)
    model_state = state.get("model")
    if model_state is not None:
        if isinstance(model_state, dict): flat = nn.utils.tree_flatten(model_state)
        else: flat = model_state
        model_param_keys = {k for k, _ in nn.utils.tree_flatten(net.parameters())}
        flat_filtered = [(k, v) for k, v in flat if k in model_param_keys]
        flat_mlx = [(k, mx.array(v)) for k, v in flat_filtered]
        tree = nn.utils.tree_unflatten(flat_mlx)
        net.update(tree)
    net.eval()
    runtime_cfg = state.get("inference_config", {"version": 1, "seed": 0, "bc_would_ko": False, "bc_wk_nvar": 10, "provenance": "none"})
    print(f"[bc-agent] loaded MLX model {_MODEL_PATH} (val_acc={state.get('val_acc', '?')})")
    return net, ckpt_cfg, runtime_cfg""")

text = text.replace("""def _build_tensors(encoded: dict, int_keys: set) -> dict:
    ob = {}
    for k, v in encoded.items():
        arr = np.asarray(v)
        if k in int_keys:
            ob[k] = torch.as_tensor(arr.astype(np.int64)).reshape(1, *arr.shape)
        else:
            ob[k] = torch.as_tensor(arr.astype(np.float16)).reshape(1, *arr.shape)
    return ob""",
"""def _build_tensors(encoded: dict, int_keys: set) -> dict:
    ob = {}
    for k, v in encoded.items():
        arr = np.asarray(v)
        if k in int_keys:
            ob[k] = mx.array(arr.astype(np.int32)).reshape(1, *arr.shape)
        else:
            ob[k] = mx.array(arr.astype(np.float16)).reshape(1, *arr.shape)
    return ob""")

text = text.replace("""def _logits_to_numpy(logits) -> np.ndarray:
    return logits.detach().to(torch.float32).numpy().flatten()""",
"""def _logits_to_numpy(logits) -> np.ndarray:
    return np.asarray(logits).flatten()""")

text = text.replace("""def _forward_baseline(ob, memory_in):
    \"\"\"Standard single forward pass.\"\"\"
    with torch.inference_mode():
        logits, _, memory_out = _LOADED_MODEL.logits_value(ob, memory_in=memory_in)
    return logits, memory_out""",
"""def _forward_baseline(ob, memory_in):
    \"\"\"Standard single forward pass.\"\"\"
    logits, _, memory_out = _LOADED_MODEL.logits_value(ob, memory_in=memory_in)
    return logits, memory_out""")

with open("scratch/stage4_native_mlx/main.py", "w") as f:
    f.write(text)
