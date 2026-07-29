"""Train/evaluate final online-GAIL components for seeds 184, 652, and 187.

Run this file directly.  Complete checkpoint folders are reused by default, so
repeating the statistical analysis does not repeat the expensive training.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import swanlab
import torch

import real_System_remake.Environment as environment_module
import real_System_remake.System as system_module
from real_System_remake.final_gail_statistical_eval import (
    DEFAULT_EXPERT_CSV,
    DEFAULT_OUTPUT_ROOT,
    evaluate_seed,
    write_summary,
)


SEEDS = (184, 652, 187)
CHECKPOINT_ROOT = (
    Path(__file__).resolve().parent / "checkpoints" / "final_weights" / "GAIL+TD3"
)
SUPPLEMENTARY_EXPERIMENT_NOTES = (
    "online GAIL+TD3 supplementary experiment: episode-level 80/20 expert split; "
    "save final production Actor and discriminator; evaluate expert, Actor, and "
    "uniform-random actions on held-out expert episodes (seeds 184, 652, 187)"
)


def checkpoint_is_complete(seed: int) -> bool:
    checkpoint_dir = CHECKPOINT_ROOT / f"seed_{seed}"
    required = (
        "production1_actor.pth",
        "discriminator.pth",
        "obs_rms_params.pth",
        "expert_split.json",
        "config.json",
    )
    return all((checkpoint_dir / name).is_file() for name in required)


def train_seed(seed: int) -> None:
    system_module.CHECKPOINT_ROOT = str(CHECKPOINT_ROOT)
    environment_module.SWANLAB_NOTES = SUPPLEMENTARY_EXPERIMENT_NOTES
    system = system_module.System(seed=seed)
    try:
        system.run()
    finally:
        if hasattr(system, "env"):
            system.env.finish()
        try:
            swanlab.finish()
        except Exception as exc:
            print(f"[SwanLab] finish warning for seed {seed}: {exc}")
        del system
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Retrain and overwrite complete checkpoints. The default reuses them.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Do not train. Fail if a required checkpoint is missing.",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        complete = checkpoint_is_complete(seed)
        if args.evaluate_only and not complete:
            raise FileNotFoundError(
                f"Incomplete checkpoint for seed {seed}: {CHECKPOINT_ROOT / f'seed_{seed}'}"
            )
        if args.force_train or not complete:
            print(f"[Train] starting seed {seed}")
            train_seed(seed)
            if not checkpoint_is_complete(seed):
                raise RuntimeError(f"Training ended without a complete checkpoint for seed {seed}.")
        else:
            print(f"[Reuse] complete checkpoint found for seed {seed}; training skipped.")

        metrics = evaluate_seed(
            seed,
            checkpoint_root=CHECKPOINT_ROOT,
            output_root=DEFAULT_OUTPUT_ROOT,
            expert_csv=DEFAULT_EXPERT_CSV,
            device_name=args.device,
        )
        print(
            f"[Evaluate] seed={seed}, actor JS={metrics['actor_vs_expert']['js_divergence']:.4f}, "
            f"actor P_eq={metrics['actor_vs_expert']['mean_equivalence_p_value']:.4e}, "
            f"random JS={metrics['random_vs_expert']['js_divergence']:.4f}, "
            f"random P_diff={metrics['random_vs_expert']['paired_permutation_p_value']:.4e}"
        )

    summary_path = write_summary(DEFAULT_OUTPUT_ROOT, SEEDS)
    print(f"[Done] summary saved to {summary_path}")


if __name__ == "__main__":
    main()
