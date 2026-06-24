from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import real_System_remake.Environment as environment_module
from Agent.TD3 import Actor


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Evaluate production-enterprise DSCR with fixed trained actors."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=base_dir / "checkpoints" / "final_weights" / "GAIL+TD3" / "seed_739",
    )
    parser.add_argument("--seed", type=int, default=739)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--algorithm", default="GAIL+TD3")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "analysis_plots" / "post_training_dscr" / "GAIL+TD3_seed_739",
    )
    return parser.parse_args()


def load_state_dict(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def actor_dimensions(state_dict: dict) -> tuple[int, int]:
    return int(state_dict["l1.weight"].shape[1]), int(state_dict["l3.weight"].shape[0])


def load_actor(path: Path, action_bound: float = 0.5) -> Actor:
    state_dict = load_state_dict(path)
    state_dim, action_dim = actor_dimensions(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = Actor(state_dim, action_dim, action_bound).to(device)
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor


def percentile_linear(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot calculate a percentile from an empty sequence.")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * probability
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index
    return (
        sorted_values[lower_index]
        + (sorted_values[upper_index] - sorted_values[lower_index]) * fraction
    )


def calculate_boxplot_statistics(values: Iterable[float]) -> tuple[dict, list[float]]:
    sorted_values = sorted(float(value) for value in values if np.isfinite(value))
    if not sorted_values:
        raise ValueError("No finite DSCR observations were collected.")

    q1 = percentile_linear(sorted_values, 0.25)
    median = percentile_linear(sorted_values, 0.50)
    q3 = percentile_linear(sorted_values, 0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    retained = [
        value for value in sorted_values
        if lower_fence <= value <= upper_fence
    ]
    statistics = {
        "count_all": len(sorted_values),
        "count_retained": len(retained),
        "count_outliers": len(sorted_values) - len(retained),
        "outlier_share_percent": 100.0 * (len(sorted_values) - len(retained)) / len(sorted_values),
        "minimum_all": sorted_values[0],
        "maximum_all": sorted_values[-1],
        "q1": q1,
        "median": median,
        "q3": q3,
        "iqr": iqr,
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
        "lower_whisker": retained[0],
        "upper_whisker": retained[-1],
        "mean_all": float(np.mean(sorted_values)),
        "mean_retained": float(np.mean(retained)),
        "median_retained": percentile_linear(retained, 0.5),
    }
    return statistics, retained


def write_rows(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_system(seed: int):
    environment_module.swanlab.init = lambda *args, **kwargs: None
    environment_module.swanlab.finish = lambda *args, **kwargs: None

    from real_System_remake.System import System

    system = System(seed=seed)
    system.env.use_swanlab = False
    system.env.logger.receive_daily_trajectory = lambda *args, **kwargs: None
    return system


def deterministic_action(actor: Actor, state: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        action = actor(np.asarray(state, dtype=np.float32).reshape(1, -1))[0]
    return action.detach().cpu().numpy()


def evaluate(args: argparse.Namespace) -> dict:
    checkpoint_dir = args.checkpoint_dir.resolve()
    required_files = [
        "production1_actor.pth",
        "consumption1_actor.pth",
        "bank1_actor.pth",
        "obs_rms_params.pth",
    ]
    missing = [name for name in required_files if not (checkpoint_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Checkpoint directory is missing required files: {', '.join(missing)}"
        )

    production_actor = load_actor(checkpoint_dir / "production1_actor.pth")
    consumption_actor = load_actor(checkpoint_dir / "consumption1_actor.pth")
    bank_actor = load_actor(checkpoint_dir / "bank1_actor.pth")
    obs_params = load_state_dict(checkpoint_dir / "obs_rms_params.pth")
    obs_mean = obs_params["mean"].detach().cpu().numpy().astype(np.float32)
    obs_var = obs_params["var"].detach().cpu().numpy().astype(np.float32)

    system = build_system(args.seed)
    daily_rows: list[dict] = []
    episode_rows: list[dict] = []
    null_output = open(os.devnull, "w", encoding="utf-8")
    try:
        for episode in range(args.episodes):
            state = system.env.reset()
            previous_dscr_count = 0
            episode_values: list[float] = []

            while True:
                production_state = np.asarray(state["production1"], dtype=np.float32)
                normalized_production_state = np.clip(
                    (production_state - obs_mean) / np.sqrt(obs_var + 1e-8),
                    -5.0,
                    5.0,
                )
                action = {
                    "production1": deterministic_action(
                        production_actor, normalized_production_state
                    ),
                    "consumption1": deterministic_action(
                        consumption_actor,
                        np.asarray(state["consumption1"], dtype=np.float32),
                    ),
                    "bank1": deterministic_action(
                        bank_actor,
                        np.asarray(state["bank1"], dtype=np.float32),
                    ) + 0.5,
                }

                with contextlib.redirect_stdout(null_output):
                    system.env.step(action)
                    next_state, _reward, done = system.env.observe()

                enterprise = system.env.Enterprise["production1"]
                if enterprise.dscr_count > previous_dscr_count:
                    dscr = float(enterprise.dscr)
                    daily_rows.append(
                        {
                            "algorithm": args.algorithm,
                            "seed": args.seed,
                            "episode": episode,
                            "day": int(system.env.day),
                            "dscr": dscr,
                            "dscr_below_1": int(dscr < 1.0),
                        }
                    )
                    episode_values.append(dscr)
                    previous_dscr_count = int(enterprise.dscr_count)

                state = next_state
                if done:
                    break

            episode_rows.append(
                {
                    "algorithm": args.algorithm,
                    "seed": args.seed,
                    "episode": episode,
                    "survival_days": int(system.env.day),
                    "valid_dscr_days": len(episode_values),
                    "episode_mean_dscr": (
                        float(np.mean(episode_values)) if episode_values else ""
                    ),
                    "episode_median_dscr": (
                        float(np.median(episode_values)) if episode_values else ""
                    ),
                    "episode_dscr_below_1_share_percent": (
                        100.0 * sum(value < 1.0 for value in episode_values) / len(episode_values)
                        if episode_values
                        else ""
                    ),
                }
            )
            if (episode + 1) % 10 == 0:
                print(f"Completed {episode + 1}/{args.episodes} episodes.")
    finally:
        null_output.close()
        system.env.finish()

    all_values = [row["dscr"] for row in daily_rows]
    boxplot_statistics, retained_values = calculate_boxplot_statistics(all_values)
    retained_set = set()
    retained_counts: dict[float, int] = {}
    for value in retained_values:
        retained_counts[value] = retained_counts.get(value, 0) + 1
    for index, row in enumerate(daily_rows):
        value = row["dscr"]
        available = retained_counts.get(value, 0)
        if available > 0:
            retained_set.add(index)
            retained_counts[value] = available - 1

    retained_rows = [
        {**row, "iqr_retained": 1}
        for index, row in enumerate(daily_rows)
        if index in retained_set
    ]
    below_one_all = sum(row["dscr"] < 1.0 for row in daily_rows)
    below_one_retained = sum(row["dscr"] < 1.0 for row in retained_rows)
    episode_signatures = {
        tuple(
            row["dscr"]
            for row in daily_rows
            if row["episode"] == episode
        )
        for episode in range(args.episodes)
    }
    unique_firm_day_pairs = {
        (row["day"], row["dscr"])
        for row in daily_rows
    }
    risk_summary = {
        "algorithm": args.algorithm,
        "seed": args.seed,
        "evaluation_episodes": args.episodes,
        "valid_firm_day_observations_all": len(daily_rows),
        "dscr_below_1_count_all": below_one_all,
        "dscr_below_1_share_percent_all": 100.0 * below_one_all / len(daily_rows),
        "valid_firm_day_observations_iqr_retained": len(retained_rows),
        "dscr_below_1_count_iqr_retained": below_one_retained,
        "dscr_below_1_share_percent_iqr_retained": (
            100.0 * below_one_retained / len(retained_rows)
        ),
        "unique_episode_trajectories": len(episode_signatures),
        "unique_day_dscr_pairs": len(unique_firm_day_pairs),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(
        args.output_dir / "production_dscr_all_firm_days.csv",
        ["algorithm", "seed", "episode", "day", "dscr", "dscr_below_1"],
        daily_rows,
    )
    write_rows(
        args.output_dir / "production_dscr_iqr_retained_firm_days.csv",
        [
            "algorithm",
            "seed",
            "episode",
            "day",
            "dscr",
            "dscr_below_1",
            "iqr_retained",
        ],
        retained_rows,
    )
    write_rows(
        args.output_dir / "production_dscr_episode_summary.csv",
        [
            "algorithm",
            "seed",
            "episode",
            "survival_days",
            "valid_dscr_days",
            "episode_mean_dscr",
            "episode_median_dscr",
            "episode_dscr_below_1_share_percent",
        ],
        episode_rows,
    )
    write_rows(
        args.output_dir / "production_dscr_boxplot_statistics.csv",
        ["algorithm", "seed", *boxplot_statistics.keys()],
        [{"algorithm": args.algorithm, "seed": args.seed, **boxplot_statistics}],
    )
    write_rows(
        args.output_dir / "production_dscr_below_1_share.csv",
        list(risk_summary.keys()),
        [risk_summary],
    )

    metadata = {
        "algorithm": args.algorithm,
        "seed": args.seed,
        "evaluation_episodes": args.episodes,
        "checkpoint_dir": str(checkpoint_dir),
        "action_policy": "deterministic actor output; no exploration noise; no learning",
        "production_state_normalization": "obs_rms_params.pth",
        "dscr_formula": "money / (should_payback + iDebt)",
        "valid_dscr_day": "day >= debt_time and should_payback + iDebt > 1e-6",
        "boxplot_rule": "Q1/Q3 linear percentiles; whiskers are observed extrema within Q1-1.5*IQR and Q3+1.5*IQR",
        "risk_share_primary_denominator": "all valid production firm-day DSCR observations",
        "unique_episode_trajectories": len(episode_signatures),
        "unique_day_dscr_pairs": len(unique_firm_day_pairs),
        "independence_warning": (
            "Repeated deterministic episodes are not independent observations."
            if len(episode_signatures) < args.episodes
            else "All episode DSCR trajectories are distinct."
        ),
    }
    (args.output_dir / "evaluation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "boxplot": boxplot_statistics,
        "risk": risk_summary,
        "output_dir": str(args.output_dir.resolve()),
    }


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
