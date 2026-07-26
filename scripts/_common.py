"""Shared helpers for the local scripts."""

import importlib.util
import logging
import os
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


def _extract_submission(tar_path: str) -> str:
    """Extract a submission.tar.gz to a temp dir and return the path."""
    tmp = tempfile.mkdtemp(prefix="ptcg_sub_")
    _temp_dirs.append(tmp)  # prevent GC
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(tmp)
    return tmp


def load_agent(path: str):
    """Import an agent from a .py file or a submission.tar.gz.

    Supports:
      - Direct .py path: agent/main.py
      - Submission tar.gz: submissions/lb881_alakazam_v1/submission.tar.gz
      - Directory with main.py: submissions/lb881_alakazam_v1/
    """
    path = os.path.abspath(path)

    # If tar.gz, extract first
    if path.endswith(".tar.gz") or path.endswith(".tgz"):
        agent_dir = _extract_submission(path)
        main_py = os.path.join(agent_dir, "main.py")
        if not os.path.exists(main_py):
            raise FileNotFoundError(f"No main.py found in {path}")
        path = main_py
    # If directory, look for main.py inside
    elif os.path.isdir(path):
        main_py = os.path.join(path, "main.py")
        if os.path.exists(main_py):
            path = main_py
        else:
            raise FileNotFoundError(f"No main.py found in {path}")

    agent_dir = os.path.dirname(path)
    spec = importlib.util.spec_from_file_location("submission_agent", path)
    module = importlib.util.module_from_spec(spec)
    # Make sibling files (deck.csv, rl/, model/) resolvable from the agent dir.
    sys.path.insert(0, agent_dir)
    # Agents may use os.path.exists("deck.csv") at module level — set CWD to agent dir.
    old_cwd = os.getcwd()
    try:
        os.chdir(agent_dir)
        spec.loader.exec_module(module)
    finally:
        os.chdir(old_cwd)
    return module.agent


def make_env():
    """Create the cabt environment (import deferred so logging is disabled)."""
    from kaggle_environments import make

    return make("cabt", configuration={})
