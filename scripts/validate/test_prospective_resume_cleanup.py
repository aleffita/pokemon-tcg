"""Validate crash-resume and post-publication cleanup on real smoke replays."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from rl.prospective_input_adapter import load_real_prospective_planner_index
from scripts.bc import build_prospective_groups as prospective


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "bc_data" / "bc_smoke_2026_07_28"
REPLAY_ZIP = ROOT / "data" / "bc_replay_zip" / "2026-07-28.zip"
CONFIG = ROOT / "configs" / "smoke.json"
SIDECAR_NAME = "prospective_publish_resume_probe"
SIDECAR = DATASET / SIDECAR_NAME
WORK = DATASET / f"{SIDECAR_NAME}.work"


class InjectedPrePublishFailure(RuntimeError):
    pass


def _build() -> dict:
    return prospective.build(
        REPLAY_ZIP,
        SIDECAR,
        config_path=CONFIG,
        max_episodes=2,
        max_groups=2,
        max_branches=4,
        trials=1,
        horizon=2,
        gamma=1.0,
        flush_groups=1,
    )


def main() -> None:
    # This namespace contains only disposable outputs of this validator.
    shutil.rmtree(SIDECAR, ignore_errors=True)
    shutil.rmtree(WORK, ignore_errors=True)

    original = prospective._summarize_branches

    def fail_before_publish(*args, **kwargs):
        raise InjectedPrePublishFailure("intentional pre-publication failure")

    prospective._summarize_branches = fail_before_publish
    try:
        try:
            _build()
        except InjectedPrePublishFailure:
            pass
        else:
            raise AssertionError("pre-publication failure was not injected")
    finally:
        prospective._summarize_branches = original

    assert not SIDECAR.exists()
    assert WORK.is_dir()
    assert len(list(WORK.glob("shard_*/.done"))) == 2

    # The first full v2 builder persisted workers=1 inside the semantic resume
    # contract. Parallel builders must accept and canonicalize that exact
    # legacy shape without discarding completed shards.
    identity_path = WORK / "build_contract.json"
    legacy_identity = json.loads(identity_path.read_text(encoding="utf-8"))
    legacy_identity["config"]["workers"] = 1
    identity_path.write_text(
        json.dumps(legacy_identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = _build()
    assert SIDECAR.is_dir()
    assert not WORK.exists()
    assert manifest["audit"]["shards_emitted"] == 2
    assert manifest["execution"]["workers"] == 2
    index = load_real_prospective_planner_index(
        DATASET,
        sidecar_name=SIDECAR_NAME,
    )
    assert len(index) == 2

    shutil.rmtree(SIDECAR)
    print(
        "[test-prospective-resume-cleanup] "
        "real pre-publish failure preserved shards; resume published and cleaned PASS"
    )


if __name__ == "__main__":
    main()
