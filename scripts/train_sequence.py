"""Build and train daily replay corpora as a bounded, resumable sequence.

Each dataset is materialized independently, trained for the configured number
of epochs, and optionally removed after the checkpoint and sequence state have
been written successfully. Raw replay ZIPs and checkpoints are retained.

Usage:
    uv run tcg-train-sequence
    uv run tcg-train-sequence --dry-run
    uv run tcg-train-sequence --config configs/train_sequence.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rl.train_config import load_config


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "train_sequence.json"


@dataclass(frozen=True)
class DatasetPhase:
    date: str
    epochs: int
    optimizer_state: str
    scheduler_state: str


def _resolve(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _parse_phases(raw: Any, default_optimizer: str, default_scheduler: str) -> list[DatasetPhase]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("datasets must be a non-empty JSON array")
    phases: list[DatasetPhase] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"datasets[{index}] must be an object")
        date = str(item.get("date", ""))
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"datasets[{index}].date must be YYYY-MM-DD, got {date!r}") from exc
        if date in seen:
            raise ValueError(f"duplicate dataset date: {date}")
        seen.add(date)
        epochs = int(item.get("epochs", 0))
        if epochs <= 0:
            raise ValueError(f"datasets[{index}].epochs must be positive")
        optimizer_state = str(item.get("optimizer_state", default_optimizer))
        scheduler_state = str(item.get("scheduler_state", default_scheduler))
        for field, value in (
            ("optimizer_state", optimizer_state),
            ("scheduler_state", scheduler_state),
        ):
            if value not in {"reset", "resume"}:
                raise ValueError(
                    f"datasets[{index}].{field} must be 'reset' or 'resume', got {value!r}"
                )
        phases.append(DatasetPhase(date, epochs, optimizer_state, scheduler_state))
    return phases


def _run(command: list[str], *, dry_run: bool) -> None:
    rendered = " ".join(command)
    print(f"\n$ {rendered}", flush=True)
    if dry_run:
        return
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"command failed with exit code {result.returncode}: {rendered}")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _safe_dataset_dir(encoded_root: Path, date: str) -> Path:
    encoded_root = encoded_root.resolve()
    dataset = (encoded_root / f"bc_{date.replace('-', '_')}").resolve()
    if dataset.parent != encoded_root or dataset == encoded_root:
        raise ValueError(f"unsafe encoded dataset path: {dataset}")
    return dataset


def _dataset_contract_issues(path: Path, expected_config) -> list[str]:
    required = (
        "__labels__.npy",
        "__would_ko_meta__.npy",
        "action_mask.npy",
        "episode_meta.npy",
        "dataset_manifest.json",
    )
    if not path.is_dir():
        return ["dataset directory is missing"]
    issues = [
        f"missing {name}" for name in required if not (path / name).is_file()
    ]
    if issues:
        return issues
    try:
        manifest = _load_json(path / "dataset_manifest.json")
    except Exception as exc:
        return [f"invalid dataset manifest: {exc}"]
    expected = {
        "bc_would_ko": bool(expected_config.bc_would_ko),
        "bc_wk_nvar": int(expected_config.bc_wk_nvar),
        "bc_both_sides": bool(expected_config.bc_both_sides),
        "seed": int(expected_config.seed),
        "max_episodes": int(expected_config.max_episodes),
        "bc_flush": int(expected_config.bc_flush),
        "self_aliases": sorted(expected_config.bc_self_aliases),
    }
    actual = manifest.get("config")
    if actual != expected:
        issues.append(
            f"manifest config {actual!r} != current build contract {expected!r}"
        )
    would_ko = manifest.get("would_ko")
    if not isinstance(would_ko, dict):
        issues.append("manifest would_ko contract is missing")
    else:
        expected_status = "computed" if expected_config.bc_would_ko else "disabled"
        if bool(would_ko.get("enabled")) != bool(expected_config.bc_would_ko):
            issues.append("manifest would_ko enabled flag differs")
        if int(would_ko.get("n_var", 0)) != int(expected_config.bc_wk_nvar):
            issues.append("manifest would_ko n_var differs")
        if str(would_ko.get("status")) != expected_status:
            issues.append(
                f"manifest would_ko status is {would_ko.get('status')!r}, "
                f"expected {expected_status!r}"
            )
    return issues


def _dataset_is_complete(path: Path, expected_config) -> bool:
    return not _dataset_contract_issues(path, expected_config)


def _checkpoint_is_valid(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 1024


def _checkpoint_epoch(path: Path) -> int:
    import pickle

    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "epoch" not in payload:
        raise ValueError(f"checkpoint has no epoch metadata: {path}")
    return int(payload["epoch"])


def _checkpoint_phase_id(path: Path) -> str | None:
    import pickle

    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint payload is not an object: {path}")
    value = payload.get("phase_id")
    return str(value) if value is not None else None


def _rolling_checkpoint(path: Path) -> Path:
    return path.with_name(f"{path.stem}_latest{path.suffix}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _phase_identity(
    date: str,
    dataset: Path,
    train_config: Path,
    *,
    dry_run: bool,
) -> tuple[str, str | None, str]:
    manifest_fingerprint: str | None = None
    manifest_path = dataset / "dataset_manifest.json"
    if manifest_path.is_file():
        manifest = _load_json(manifest_path)
        value = manifest.get("build_fingerprint")
        manifest_fingerprint = str(value) if value is not None else None
    elif not dry_run:
        raise RuntimeError(f"dataset manifest is missing: {manifest_path}")
    train_config_sha256 = _sha256_file(train_config)
    payload = {
        "date": date,
        "dataset_build_fingerprint": manifest_fingerprint,
        "train_config_sha256": train_config_sha256,
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), manifest_fingerprint, train_config_sha256


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, tmp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        shutil.copy2(source, tmp_name)
        with open(tmp_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp_name, destination)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _free_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024**3)


def _remove_reproducible_dataset(path: Path, zip_path: Path) -> None:
    if not zip_path.is_file():
        raise RuntimeError(f"refusing to remove {path}: source replay ZIP is missing: {zip_path}")
    if path.is_dir():
        size_gib = sum(
            entry.stat().st_size for entry in path.rglob("*") if entry.is_file()
        ) / (1024**3)
        shutil.rmtree(path)
        print(
            f"Removed reproducible encoded dataset {path} ({size_gib:.1f} GiB); "
            f"source retained at {zip_path}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore prior sequence state. Existing output checkpoint is never deleted.",
    )
    args = parser.parse_args()

    config_path = _resolve(args.config)
    cfg = _load_json(config_path)

    train_config = _resolve(cfg.get("train_config", "configs/train_config.json"))
    replay_root = _resolve(cfg.get("replay_zip_dir", "data/bc_replay_zip"))
    encoded_root = _resolve(cfg.get("encoded_root", "data/bc_data"))
    checkpoint = _resolve(
        cfg.get("checkpoint", "model/checkpoint/bc_temporal_v2_mlx.pkl")
    )
    state_path = _resolve(
        cfg.get("state_file", "model/checkpoint/train_sequence_state.json")
    )
    initial_checkpoint_raw = cfg.get("initial_checkpoint")
    initial_checkpoint = (
        _resolve(initial_checkpoint_raw) if initial_checkpoint_raw else None
    )
    fresh_start = bool(cfg.get("fresh_start", initial_checkpoint is None))
    if fresh_start and initial_checkpoint is not None:
        raise ValueError("fresh_start=true conflicts with initial_checkpoint")

    default_optimizer = str(cfg.get("optimizer_state", "reset"))
    default_scheduler = str(cfg.get("scheduler_state", "reset"))
    phases = _parse_phases(
        cfg.get("datasets"), default_optimizer, default_scheduler
    )
    build = cfg.get("build", {})
    if not isinstance(build, dict):
        raise ValueError("build must be a JSON object")
    build_overrides = {}
    for key, field in (
        ("workers", "bc_workers"),
        ("flush", "bc_flush"),
        ("ep_timeout", "bc_ep_timeout"),
        ("max_episodes", "max_episodes"),
    ):
        if key in build:
            build_overrides[field] = build[key]
    expected_dataset_config = load_config(
        cli_overrides=build_overrides,
        config_path=str(train_config),
    )

    download_missing = bool(cfg.get("download_missing", True))
    rebuild_encoded = bool(cfg.get("rebuild_encoded", True))
    delete_encoded = bool(cfg.get("delete_encoded_after_success", True))
    min_free_gib = float(cfg.get("min_free_gib", 8.0))

    if not train_config.is_file():
        raise FileNotFoundError(f"training config not found: {train_config}")
    replay_root.mkdir(parents=True, exist_ok=True)
    encoded_root.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    previous_state: dict[str, Any] = {}
    if state_path.is_file() and not args.restart:
        previous_state = _load_json(state_path)
    completed = set(previous_state.get("completed_dates", []))
    active_phase = previous_state.get("active_phase")
    if active_phase is not None and not isinstance(active_phase, dict):
        raise ValueError(f"invalid active_phase in {state_path}")
    resume_checkpoint: Path | None
    if completed:
        saved_checkpoint = _resolve(previous_state.get("checkpoint", str(checkpoint)))
        if not _checkpoint_is_valid(saved_checkpoint):
            raise RuntimeError(
                f"sequence state lists completed datasets but checkpoint is invalid: "
                f"{saved_checkpoint}"
            )
        resume_checkpoint = saved_checkpoint
    elif initial_checkpoint is not None:
        if not _checkpoint_is_valid(initial_checkpoint):
            raise FileNotFoundError(f"initial checkpoint is invalid: {initial_checkpoint}")
        resume_checkpoint = initial_checkpoint
    else:
        resume_checkpoint = None
    simulated_checkpoint_epoch = (
        _checkpoint_epoch(resume_checkpoint)
        if resume_checkpoint is not None
        else -1
    )

    rolling_checkpoint = _rolling_checkpoint(checkpoint)
    if (
        fresh_start
        and not previous_state
        and (
            _checkpoint_is_valid(checkpoint)
            or _checkpoint_is_valid(rolling_checkpoint)
        )
    ):
        raise RuntimeError(
            f"fresh sequence has no state file but output checkpoint already "
            f"exists at {checkpoint} or {rolling_checkpoint}; choose a new "
            "checkpoint path or restore "
            "the matching sequence state before continuing"
        )

    print("Daily BC training sequence", flush=True)
    print(f"  config:      {config_path}", flush=True)
    print(f"  train config:{train_config}", flush=True)
    print(f"  checkpoint:  {checkpoint}", flush=True)
    print(f"  phases:      {len(phases)}", flush=True)
    print(f"  completed:   {sorted(completed)}", flush=True)

    for phase in phases:
        if phase.date in completed:
            print(f"\n[skip] {phase.date}: already completed", flush=True)
            continue
        if active_phase is not None and active_phase.get("date") != phase.date:
            raise RuntimeError(
                f"sequence state has unfinished phase {active_phase.get('date')!r}; "
                f"refusing to start {phase.date!r}"
            )

        zip_path = replay_root / f"{phase.date}.zip"
        dataset = _safe_dataset_dir(encoded_root, phase.date)

        if not zip_path.is_file():
            if not download_missing:
                raise FileNotFoundError(f"replay ZIP not found: {zip_path}")
            _run(
                ["uv", "run", "tcg-data", "--date", phase.date],
                dry_run=args.dry_run,
            )
        if not args.dry_run and not zip_path.is_file():
            raise RuntimeError(f"download completed without producing {zip_path}")

        contract_issues = _dataset_contract_issues(
            dataset, expected_dataset_config
        )
        if (
            dataset.exists()
            and active_phase is None
            and (rebuild_encoded or contract_issues)
        ):
            if args.dry_run:
                reason = (
                    "forced rebuild"
                    if rebuild_encoded
                    else "; ".join(contract_issues)
                )
                print(
                    f"[dry-run] would remove reproducible encoded dataset "
                    f"{dataset}: {reason}"
                )
            else:
                _remove_reproducible_dataset(dataset, zip_path)

        if not _dataset_is_complete(dataset, expected_dataset_config):
            free_gib = _free_gib(encoded_root)
            if free_gib < min_free_gib:
                raise RuntimeError(
                    f"only {free_gib:.1f} GiB free before build; "
                    f"minimum configured is {min_free_gib:.1f} GiB"
                )
            command = [
                "uv",
                "run",
                "tcg-build-bc",
                str(dataset),
                str(zip_path),
                "--config",
                str(train_config),
            ]
            for key, flag in (
                ("workers", "--workers"),
                ("flush", "--flush"),
                ("ep_timeout", "--ep-timeout"),
                ("max_episodes", "--max-episodes"),
            ):
                if key in build:
                    command.extend([flag, str(build[key])])
            _run(command, dry_run=args.dry_run)
        if not args.dry_run and not _dataset_is_complete(
            dataset, expected_dataset_config
        ):
            raise RuntimeError(
                f"dataset build did not produce the complete contract at {dataset}"
            )

        (
            phase_id,
            dataset_build_fingerprint,
            train_config_sha256,
        ) = _phase_identity(
            phase.date,
            dataset,
            train_config,
            dry_run=args.dry_run,
        )

        phase_optimizer_state = phase.optimizer_state
        phase_scheduler_state = phase.scheduler_state
        phase_resume = resume_checkpoint
        phase_epochs = phase.epochs
        if active_phase is not None:
            if int(active_phase.get("target_epochs", -1)) != phase.epochs:
                raise RuntimeError(
                    f"active phase target changed for {phase.date}: "
                    f"state={active_phase.get('target_epochs')} config={phase.epochs}"
                )
            phase_start_epoch = int(active_phase["start_epoch"])
            if str(active_phase.get("phase_id")) != phase_id:
                raise RuntimeError(
                    f"active phase contract changed for {phase.date}; "
                    "dataset or train config differs from the recorded phase"
                )
            rolling = _rolling_checkpoint(checkpoint)
            if (
                _checkpoint_is_valid(rolling)
                and _checkpoint_phase_id(rolling) == phase_id
            ):
                finished_epochs = _checkpoint_epoch(rolling) - phase_start_epoch + 1
                if finished_epochs < 0 or finished_epochs > phase.epochs:
                    raise RuntimeError(
                        f"rolling checkpoint epoch is outside active phase: "
                        f"start={phase_start_epoch}, completed={finished_epochs}, "
                        f"target={phase.epochs}"
                    )
                if finished_epochs > 0:
                    phase_epochs = phase.epochs - finished_epochs
                    phase_resume = rolling
                    phase_optimizer_state = "resume"
                    phase_scheduler_state = "resume"
                    print(
                        f"[resume] {phase.date}: {finished_epochs}/{phase.epochs} "
                        f"epochs complete; continuing {phase_epochs} from {rolling}",
                        flush=True,
                    )
            else:
                finished_epochs = 0
            if finished_epochs == 0:
                base_checkpoint_raw = active_phase.get("base_checkpoint")
                phase_resume = (
                    _resolve(base_checkpoint_raw)
                    if base_checkpoint_raw
                    else None
                )
                if (
                    phase_resume is not None
                    and not _checkpoint_is_valid(phase_resume)
                ):
                    raise RuntimeError(
                        f"active phase base checkpoint is invalid: {phase_resume}"
                    )
                phase_optimizer_state = str(active_phase["optimizer_state"])
                phase_scheduler_state = str(active_phase["scheduler_state"])
                phase_epochs = phase.epochs
                print(
                    f"[resume] {phase.date}: no completed epoch checkpoint; "
                    "restarting the phase from its recorded base",
                    flush=True,
                )
        else:
            if (
                args.dry_run
                and phase_resume == checkpoint
                and not _checkpoint_is_valid(checkpoint)
            ):
                phase_start_epoch = simulated_checkpoint_epoch + 1
            else:
                phase_start_epoch = (
                    _checkpoint_epoch(phase_resume) + 1
                    if phase_resume is not None
                    else 0
                )
            active_phase = {
                "date": phase.date,
                "target_epochs": phase.epochs,
                "start_epoch": phase_start_epoch,
                "optimizer_state": phase.optimizer_state,
                "scheduler_state": phase.scheduler_state,
                "base_checkpoint": (
                    str(phase_resume) if phase_resume is not None else None
                ),
                "phase_id": phase_id,
                "dataset_build_fingerprint": dataset_build_fingerprint,
                "train_config_sha256": train_config_sha256,
                "training_started": False,
            }

        if not args.dry_run:
            state = {
                "schema_version": 1,
                "config": str(config_path),
                "train_config": str(train_config),
                "checkpoint": str(checkpoint),
                "completed_dates": [
                    item.date for item in phases if item.date in completed
                ],
                "active_phase": {**active_phase, "training_started": True},
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            _atomic_json(state_path, state)
            active_phase = state["active_phase"]

        if phase_epochs <= 0:
            print(
                f"[resume] {phase.date}: all configured epochs are already checkpointed",
                flush=True,
            )
        train_command = [
            "uv",
            "run",
            "tcg-train",
            str(dataset),
            "--config",
            str(train_config),
            "--epochs",
            str(phase_epochs),
            "--out",
            str(checkpoint),
            "--optimizer-state",
            phase_optimizer_state,
            "--scheduler-state",
            phase_scheduler_state,
            "--phase-id",
            phase_id,
        ]
        if phase_resume is not None:
            train_command.extend(["--resume", str(phase_resume)])
        if phase_epochs > 0:
            _run(train_command, dry_run=args.dry_run)

        if args.dry_run:
            simulated_checkpoint_epoch = (
                phase_start_epoch + phase.epochs - 1
            )
            resume_checkpoint = checkpoint
            active_phase = None
            continue
        if not _checkpoint_is_valid(checkpoint):
            raise RuntimeError(f"trainer did not produce a valid checkpoint: {checkpoint}")
        rolling = _rolling_checkpoint(checkpoint)
        if (
            not _checkpoint_is_valid(rolling)
            or _checkpoint_phase_id(rolling) != phase_id
            or _checkpoint_epoch(rolling) < phase_start_epoch + phase.epochs - 1
        ):
            raise RuntimeError(
                f"phase did not reach its target rolling checkpoint: {rolling}"
            )

        # Across daily corpora, validation scores are not directly comparable.
        # The sequence's canonical checkpoint is therefore the last completed
        # state, while the trainer's per-phase best remains available in its
        # numbered/best artifacts.
        _atomic_copy(rolling, checkpoint)

        completed.add(phase.date)
        state = {
            "schema_version": 1,
            "config": str(config_path),
            "train_config": str(train_config),
            "checkpoint": str(checkpoint),
            "last_phase_id": phase_id,
            "completed_dates": [item.date for item in phases if item.date in completed],
            "active_phase": None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(state_path, state)
        active_phase = None
        resume_checkpoint = checkpoint

        if delete_encoded:
            _remove_reproducible_dataset(dataset, zip_path)

    print("\nSequence complete.", flush=True)


if __name__ == "__main__":
    main()
