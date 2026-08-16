"""Run repeated fresh-rollout GRPO cycles within one wall-clock budget."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import time
from typing import Any, Callable

from scripts.rl.run_ar021 import run_ar021


DEFAULT_OPPONENT_DECKS = [
    Path("public_agents/lb1009_mega_lucario_ex_islet/deck.csv"),
    Path("public_agents/lb945_multiply_ivan/deck.csv"),
    Path("public_agents/lb826_alakazam_seok/deck.csv"),
    Path("public_agents/lb814_crustle_emre/deck.csv"),
]
DEFAULT_OPPONENT_AGENTS = [path.with_name("main.py") for path in DEFAULT_OPPONENT_DECKS]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    incomplete = path.with_name(f"{path.name}.incomplete")
    incomplete.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    incomplete.replace(path)


def _append_event(path: Path, event: dict[str, Any]) -> None:
    payload = {
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        **event,
    }
    with path.open("a") as stream:
        stream.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
    print(f"[continuous-grpo] {json.dumps(payload, sort_keys=True)}", flush=True)


def run_continuous_grpo(
    *,
    initial_checkpoint: Path,
    output_dir: Path,
    deck_path: Path,
    meta_date: str,
    opponent_deck_paths: list[Path],
    opponent_agent_paths: list[Path],
    total_budget_seconds: float = 46_800.0,
    cycle_update_seconds: float = 720.0,
    update_epochs_per_cycle: int = 3,
    generated_population_size: int = 4,
    minimum_cycle_seconds: float = 240.0,
    finalization_reserve_seconds: float = 90.0,
    groups_per_matchup: int = 1,
    learning_rate: float = 2e-7,
    seed: int = 38_038,
    max_cycles: int | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Refresh behavior data and model-generated decks after every short update."""
    if total_budget_seconds <= 0.0:
        raise ValueError("total_budget_seconds must be positive")
    if cycle_update_seconds <= 0.0:
        raise ValueError("cycle_update_seconds must be positive")
    if update_epochs_per_cycle < 1:
        raise ValueError("update_epochs_per_cycle must be at least one")
    if generated_population_size < 0:
        raise ValueError("generated_population_size must be non-negative")
    if len(opponent_deck_paths) != len(opponent_agent_paths):
        raise ValueError("opponent deck and agent lists must align")

    output_dir.mkdir(parents=True, exist_ok=True)
    event_path = output_dir / "continuous.events.jsonl"
    event_path.write_text("")
    started = clock()
    deadline = started + total_budget_seconds
    parent = initial_checkpoint
    generated_population: list[Path] = []
    completed_cycles: list[dict[str, Any]] = []
    cycle_index = 0
    _append_event(
        event_path,
        {
            "event": "continuous_start",
            "initial_checkpoint": str(initial_checkpoint),
            "total_budget_seconds": total_budget_seconds,
            "update_epochs_per_cycle": update_epochs_per_cycle,
            "generated_population_size": generated_population_size,
        },
    )

    while max_cycles is None or cycle_index < max_cycles:
        remaining = deadline - clock()
        if remaining < minimum_cycle_seconds:
            break
        cycle_index += 1
        cycle_name = f"AR-038-C{cycle_index:03d}"
        cycle_dir = output_dir / f"cycle-{cycle_index:03d}"
        available_update_seconds = max(1.0, remaining - finalization_reserve_seconds)
        update_budget = min(cycle_update_seconds, available_update_seconds)
        learner_decks = [deck_path, *generated_population]
        _append_event(
            event_path,
            {
                "event": "cycle_start",
                "cycle": cycle_index,
                "experiment": cycle_name,
                "parent": str(parent),
                "remaining_seconds": round(remaining, 3),
                "cycle_update_seconds": update_budget,
                "generated_population": [str(path) for path in generated_population],
            },
        )
        try:
            manifest = run_ar021(
                checkpoint=parent,
                deck_path=deck_path,
                learner_deck_paths=learner_decks,
                meta_date=meta_date,
                output_dir=cycle_dir,
                opponent_deck_paths=opponent_deck_paths,
                opponent_agent_paths=opponent_agent_paths,
                groups_per_matchup=groups_per_matchup,
                k_max=4,
                branch_uniform_mix=0.5,
                update_epochs=update_epochs_per_cycle,
                deck_relative_weight=0.5,
                learning_rate=learning_rate,
                ropend=True,
                prospective_aux_weight=0.2,
                deck_aux_weight=0.1,
                aux_batch_size=512,
                policy_group_batch_size=2,
                max_update_seconds=update_budget,
                deck_action_weight=0.25,
                dense_reward_weight=0.49,
                include_generated_deck=True,
                deck_pool_dir=Path("experiments/decks/swarm/inbox"),
                deck_pool_limit=0,
                swarm_results_dir=Path("experiments/decks/swarm/results"),
                update_device="cpu",
                seed=seed + cycle_index * 100_000,
                experiment=cycle_name,
            )
        except Exception as exc:
            _append_event(
                event_path,
                {
                    "event": "cycle_failed",
                    "cycle": cycle_index,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise

        parent = Path(manifest["candidate"])
        generated = cycle_dir / "decks" / "generated_turn0.json"
        if generated_population_size > 0 and generated.is_file():
            generated_population.append(generated)
            generated_population = generated_population[-generated_population_size:]
        summary = {
            "cycle": cycle_index,
            "experiment": cycle_name,
            "candidate": str(parent),
            "candidate_sha256": manifest["candidate_sha256"],
            "games": manifest["group_size"],
            "logical_decisions": manifest["logical_decisions"],
            "optimizer_steps": manifest["metrics"]["optimizer_steps"],
            "generated_deck": str(generated) if generated.is_file() else None,
            "elapsed_seconds": round(clock() - started, 3),
        }
        completed_cycles.append(summary)
        _atomic_json(
            output_dir / "latest.json",
            {
                "format": "ptcg-continuous-grpo-v1",
                "initial_checkpoint": str(initial_checkpoint),
                "latest": summary,
                "generated_population": [str(path) for path in generated_population],
                "completed_cycle_count": len(completed_cycles),
                "total_budget_seconds": total_budget_seconds,
            },
        )
        _append_event(event_path, {"event": "cycle_complete", **summary})

    result = {
        "format": "ptcg-continuous-grpo-v1",
        "initial_checkpoint": str(initial_checkpoint),
        "final_checkpoint": str(parent),
        "completed_cycle_count": len(completed_cycles),
        "cycles": completed_cycles,
        "generated_population": [str(path) for path in generated_population],
        "elapsed_seconds": round(clock() - started, 3),
        "total_budget_seconds": total_budget_seconds,
    }
    _atomic_json(output_dir / "manifest.json", result)
    _append_event(event_path, {"event": "continuous_complete", **result})
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--agent-deck", type=Path, required=True)
    parser.add_argument("--meta-date", required=True)
    parser.add_argument("--total-budget-seconds", type=float, default=46_800.0)
    parser.add_argument("--cycle-update-seconds", type=float, default=720.0)
    parser.add_argument("--update-epochs-per-cycle", type=int, default=3)
    parser.add_argument("--generated-population-size", type=int, default=4)
    parser.add_argument("--groups-per-matchup", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-7)
    parser.add_argument("--seed", type=int, default=38_038)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_continuous_grpo(
        initial_checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        deck_path=args.agent_deck,
        meta_date=args.meta_date,
        opponent_deck_paths=DEFAULT_OPPONENT_DECKS,
        opponent_agent_paths=DEFAULT_OPPONENT_AGENTS,
        total_budget_seconds=args.total_budget_seconds,
        cycle_update_seconds=args.cycle_update_seconds,
        update_epochs_per_cycle=args.update_epochs_per_cycle,
        generated_population_size=args.generated_population_size,
        groups_per_matchup=args.groups_per_matchup,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
