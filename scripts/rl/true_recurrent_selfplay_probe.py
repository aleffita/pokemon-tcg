"""Small executable current-vs-current true recurrent self-play probe."""

from __future__ import annotations

import json

from scripts.rl.trajectory_probe import build_true_recurrent_parser, run_true_recurrent_probe


def main() -> None:
    args = build_true_recurrent_parser().parse_args()
    manifest = run_true_recurrent_probe(
        checkpoint=args.checkpoint,
        deck_path=args.agent_deck,
        meta_date=args.meta_date,
        output_dir=args.output_dir,
        games=args.games,
        seed=args.seed,
        experiment=args.experiment,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
