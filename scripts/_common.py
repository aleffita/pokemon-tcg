"""Shared helpers for the local scripts."""

import atexit
import importlib.util
import logging
import os
import shutil
import sys
import tarfile
import tempfile

# Silence noisy kaggle_environments logs and warnings from unused environments.
logging.disable(logging.CRITICAL)

# Pre-import kaggle_environments with stdout/stderr suppressed at the OS level.
_devnull = os.open(os.devnull, os.O_WRONLY)
_old_stdout = os.dup(1)
_old_stderr = os.dup(2)
os.dup2(_devnull, 1)
os.dup2(_devnull, 2)
try:
    import warnings
    warnings.filterwarnings("ignore")
    import kaggle_environments  # noqa: F401 — pre-cache, suppress warnings
finally:
    os.dup2(_old_stdout, 1)
    os.dup2(_old_stderr, 2)
    os.close(_old_stdout)
    os.close(_old_stderr)
    os.close(_devnull)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT_DIR = os.path.join(ROOT, "agent")

# Keep references to temp dirs so they aren't garbage-collected
_temp_dirs: list[str] = []
_submission_cache: dict[str, str] = {}
_agent_sys_paths: set[str] = set()


def _cleanup_temp_dirs() -> None:
    for path in _temp_dirs:
        shutil.rmtree(path, ignore_errors=True)


atexit.register(_cleanup_temp_dirs)


def _extract_submission(tar_path: str) -> str:
    """Extract a submission once per process and return its temporary path."""
    tar_path = os.path.realpath(tar_path)
    cached = _submission_cache.get(tar_path)
    if cached and os.path.isdir(cached):
        return cached
    tmp = tempfile.mkdtemp(prefix="ptcg_sub_")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(tmp)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    _temp_dirs.append(tmp)
    _submission_cache[tar_path] = tmp
    return tmp


def materialize_agent_path(path: str) -> str:
    """Return a stable main.py path, extracting packaged agents only once."""
    absolute = os.path.abspath(path)
    if absolute.endswith((".tar.gz", ".tgz")):
        main_py = os.path.join(_extract_submission(absolute), "main.py")
        if not os.path.exists(main_py):
            raise FileNotFoundError(f"No main.py found in {absolute}")
        return main_py
    return absolute


def load_agent(path: str, return_module: bool = False):
    """Import an agent from a .py file or a submission.tar.gz.

    Supports:
      - Direct .py path: agent/main.py
      - Submission tar.gz: submissions/lb881_alakazam_v1/submission.tar.gz
      - Directory with main.py: submissions/lb881_alakazam_v1/

    Returns the agent callable, or (callable, module) when `return_module` is
    set — callers that swap deck.csv between runs need the module to reload it.
    """
    path = materialize_agent_path(path)

    # If tar.gz, extract first
    if os.path.isdir(path):
        sub_tar = os.path.join(path, "submission.tar.gz")
        main_py = os.path.join(path, "main.py")
        sub_main_py = os.path.join(path, "submission", "main.py")
        if os.path.exists(main_py):
            path = main_py
        elif os.path.exists(sub_tar):
            agent_dir = _extract_submission(sub_tar)
            path = os.path.join(agent_dir, "main.py")
        elif os.path.exists(sub_main_py):
            path = sub_main_py
        else:
            raise FileNotFoundError(f"No main.py or submission.tar.gz found in {path}")

    agent_dir = os.path.dirname(path)
    # Auto-detect backend: if tarball has MLX checkpoint but no PyTorch .pt file, set PTCG_INFERENCE_BACKEND=mlx
    has_mlx = os.path.exists(os.path.join(agent_dir, "model", "bc_model", "bc_best_mlx_final.pkl"))
    has_pt = (
        os.path.exists(os.path.join(agent_dir, "model", "bc_model", "bc_best_torch_fp16.pt"))
        or os.path.exists(os.path.join(agent_dir, "model", "bc_best_final.pt"))
    )
    if has_mlx and not has_pt:
        os.environ["PTCG_INFERENCE_BACKEND"] = "mlx"
    elif has_pt:
        os.environ["PTCG_INFERENCE_BACKEND"] = "torch"

    # Purge previously loaded submission agent/rl modules from sys.modules to prevent cross-agent backend cache pollution
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("rl.") or mod_name in ("submission_agent", "rl"):
            del sys.modules[mod_name]

    spec = importlib.util.spec_from_file_location("submission_agent", path)
    module = importlib.util.module_from_spec(spec)
    # Make sibling files (deck.csv, rl/, model/) resolvable from the agent dir.
    public_agents_root = os.path.join(ROOT, "public_agents")
    sys.path[:] = [
        entry
        for entry in sys.path
        if entry not in _agent_sys_paths
        and "ptcg_sub_" not in entry
        and not os.path.realpath(entry).startswith(public_agents_root)
    ]
    sys.path.insert(0, agent_dir)
    _agent_sys_paths.add(agent_dir)
    # Agents may use os.path.exists("deck.csv") at module level — set CWD to agent dir.
    old_cwd = os.getcwd()
    try:
        os.chdir(agent_dir)
        spec.loader.exec_module(module)
    finally:
        os.chdir(old_cwd)
    return (module.agent, module) if return_module else module.agent


def make_env():
    """Create the cabt environment (import deferred so logging is disabled)."""
    from kaggle_environments import make

    return make("cabt", configuration={})
