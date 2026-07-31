"""Retrain three seeds with Actor snapshots and run strict phase-2 evaluation.

Run directly with no arguments. Existing checkpoints are reused only when a
non-empty Actor-snapshot manifest and every referenced snapshot are present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from real_System_remake.evaluate_actor_snapshot_evolution import (
    DEFAULT_EXPERT_CSV,
    DEFAULT_OUTPUT_DIR,
    evaluate_all,
)
from real_System_remake.run_three_seed_final_gail_eval import (
    CHECKPOINT_ROOT,
    MINIMUM_ACTOR_SNAPSHOT_COUNT,
    SEEDS,
    train_seed,
)


def snapshots_are_complete(seed: int) -> bool:
    snapshot_dir = CHECKPOINT_ROOT / f"seed_{seed}" / "actor_snapshots"
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        snapshots = manifest.get("snapshots", [])
    except (OSError, ValueError, TypeError):
        return False
    evaluation_steps = [int(item["evaluation_step"]) for item in snapshots]
    return (
        len(snapshots) >= MINIMUM_ACTOR_SNAPSHOT_COUNT
        and evaluation_steps == list(range(1, len(snapshots) + 1))
        and all(
            (snapshot_dir / item["actor_checkpoint"]).is_file()
            for item in snapshots
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Retrain all three seeds even when complete Actor snapshots exist.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Evaluate existing snapshots and fail instead of retraining when absent.",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        complete = snapshots_are_complete(seed)
        if args.evaluate_only and not complete:
            raise FileNotFoundError(
                f"Seed {seed} lacks complete Actor snapshots. Retraining is required."
            )
        if args.force_train or not complete:
            print(f"[Snapshot training] starting seed {seed}")
            train_seed(seed)
            if not snapshots_are_complete(seed):
                raise RuntimeError(
                    f"Seed {seed} training ended without complete Actor snapshots."
                )
        else:
            print(f"[Snapshot reuse] seed {seed} has complete Actor snapshots")

    output_dir = evaluate_all(
        SEEDS,
        checkpoint_root=CHECKPOINT_ROOT,
        expert_csv=DEFAULT_EXPERT_CSV,
        output_dir=DEFAULT_OUTPUT_DIR,
        device_name=args.device,
    )
    print(f"[Done] strict Actor-evolution results saved to: {output_dir}")


if __name__ == "__main__":
    main()
