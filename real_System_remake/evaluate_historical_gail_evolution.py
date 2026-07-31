"""Retrospective phase-2 evaluation of online GAIL training trajectories.

For each seed, the discriminator selected by validation is frozen and used as
a common ruler for every historical production-enterprise state-action pair.
The script compares each episode's discriminator-score distribution with the
held-out expert-test score distribution using Jensen-Shannon divergence.

The logged action is the behavior action actually applied during training.  It
therefore contains both the Actor output and the exploration perturbation used
by TD3.  Historical deterministic Actor outputs cannot be reconstructed unless
an Actor checkpoint was saved for every episode.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.stats import t as student_t
import torch

try:
    from real_System_remake.expert_data_split import (
        file_sha256,
        load_expert_episode_split,
    )
    from real_System_remake.final_gail_statistical_eval import (
        FinalDiscriminator,
        _load_rms,
        _load_state_dict,
    )
except ModuleNotFoundError:
    from expert_data_split import file_sha256, load_expert_episode_split
    from final_gail_statistical_eval import (
        FinalDiscriminator,
        _load_rms,
        _load_state_dict,
    )


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_ROOT = (
    PROJECT_DIR / "checkpoints" / "final_weights" / "GAIL+TD3_balanced_discriminator"
)
DEFAULT_TRAJECTORY_ROOT = PROJECT_DIR / "trajectory_logs"
DEFAULT_EXPERT_CSV = PROJECT_DIR / "expert_data_production1_collected.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "analysis_plots" / "historical_gail_evolution"
DEFAULT_SEEDS = (184, 652, 187)

STATE_COLUMNS = [f"state_{index}" for index in range(33)]
ACTION_COLUMNS = [f"action_{index}" for index in range(4)]
TRAJECTORY_COLUMNS = ["episode", "day", "done", *STATE_COLUMNS, *ACTION_COLUMNS]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _score_pairs(
    discriminator: FinalDiscriminator,
    states: np.ndarray,
    actions: np.ndarray,
    rms: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    scores: list[np.ndarray] = []
    discriminator.eval()
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            end = min(start + batch_size, len(states))
            state = torch.as_tensor(states[start:end], dtype=torch.float32, device=device)
            action = torch.as_tensor(actions[start:end], dtype=torch.float32, device=device)
            state = torch.clamp(
                (state - rms["mean"]) / torch.sqrt(rms["var"] + 1e-8),
                -5.0,
                5.0,
            )
            action = (action - rms["act_mean"]) / torch.sqrt(rms["act_var"] + 1e-8)
            scores.append(torch.sigmoid(discriminator(state, action)).cpu().numpy().reshape(-1))
    return np.concatenate(scores).astype(np.float64, copy=False)


def _js_from_histograms(reference: np.ndarray, target: np.ndarray) -> float:
    reference_prob = reference.astype(np.float64)
    target_prob = target.astype(np.float64)
    reference_prob /= max(float(reference_prob.sum()), 1.0)
    target_prob /= max(float(target_prob.sum()), 1.0)
    return float(distance.jensenshannon(reference_prob, target_prob) ** 2)


def _load_expert_reference(
    expert_csv: Path,
    split_manifest_path: Path,
    discriminator: FinalDiscriminator,
    rms: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
    bins: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    split = load_expert_episode_split(expert_csv)
    checkpoint_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    if split.metadata["expert_csv_sha256"] != checkpoint_manifest["expert_csv_sha256"]:
        raise ValueError(
            "The current expert CSV does not match the expert data used by the checkpoint."
        )
    expert_values = split.test_values
    expert_scores = _score_pairs(
        discriminator,
        expert_values[:, :33],
        expert_values[:, 33:37],
        rms,
        device,
        batch_size,
    )
    edges = np.linspace(0.0, 1.0, bins + 1)
    expert_histogram, _ = np.histogram(expert_scores, bins=edges)
    metadata = {
        "partition": "held_out_final_test_complete_expert_episodes",
        "row_count": int(expert_scores.size),
        "episode_count": int(np.unique(split.test_episode_labels).size),
        "score_mean": float(expert_scores.mean()),
        "score_std": float(expert_scores.std(ddof=1)),
        "expert_csv_sha256": split.metadata["expert_csv_sha256"],
    }
    return expert_scores, expert_histogram, metadata


def _read_and_score_trajectory(
    trajectory_path: Path,
    discriminator: FinalDiscriminator,
    rms: dict[str, torch.Tensor],
    expert_histogram: np.ndarray,
    device: torch.device,
    *,
    bins: int,
    chunk_size: int,
    score_batch_size: int,
    rolling_episodes: int,
    window_episodes: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    episode_histograms: dict[int, np.ndarray] = {}
    episode_score_sums: dict[int, float] = {}
    episode_counts: dict[int, int] = {}
    episode_last_days: dict[int, int] = {}
    episode_done: dict[int, int] = {}
    nonfinite_rows = 0
    total_rows = 0

    dtype = {"episode": "int64", "day": "int32", "done": "int8"}
    dtype.update({column: "float32" for column in STATE_COLUMNS + ACTION_COLUMNS})
    reader = pd.read_csv(
        trajectory_path,
        usecols=TRAJECTORY_COLUMNS,
        dtype=dtype,
        chunksize=chunk_size,
    )
    for chunk_index, chunk in enumerate(reader, start=1):
        states = chunk[STATE_COLUMNS].to_numpy(dtype=np.float32, copy=False)
        actions = chunk[ACTION_COLUMNS].to_numpy(dtype=np.float32, copy=False)
        finite = np.isfinite(states).all(axis=1) & np.isfinite(actions).all(axis=1)
        nonfinite_rows += int((~finite).sum())
        if not finite.any():
            continue
        chunk = chunk.loc[finite]
        states = states[finite]
        actions = actions[finite]
        scores = _score_pairs(
            discriminator,
            states,
            actions,
            rms,
            device,
            score_batch_size,
        )
        episodes = chunk["episode"].to_numpy(dtype=np.int64, copy=False)
        days = chunk["day"].to_numpy(dtype=np.int32, copy=False)
        done_values = chunk["done"].to_numpy(dtype=np.int8, copy=False)
        total_rows += int(scores.size)
        for episode in np.unique(episodes):
            mask = episodes == episode
            histogram, _ = np.histogram(scores[mask], bins=edges)
            episode_id = int(episode)
            if episode_id not in episode_histograms:
                episode_histograms[episode_id] = np.zeros(bins, dtype=np.int64)
                episode_score_sums[episode_id] = 0.0
                episode_counts[episode_id] = 0
                episode_last_days[episode_id] = 0
                episode_done[episode_id] = 0
            episode_histograms[episode_id] += histogram
            episode_score_sums[episode_id] += float(scores[mask].sum())
            episode_counts[episode_id] += int(mask.sum())
            episode_last_days[episode_id] = max(
                episode_last_days[episode_id], int(days[mask].max())
            )
            episode_done[episode_id] = max(
                episode_done[episode_id], int(done_values[mask].max())
            )
        if chunk_index % 10 == 0:
            print(
                f"  {trajectory_path.parent.name}: scored {total_rows:,} rows "
                f"across {len(episode_histograms):,} episodes"
            )

    episode_rows: list[dict[str, Any]] = []
    for sequence_index, episode in enumerate(sorted(episode_histograms), start=1):
        count = episode_counts[episode]
        episode_rows.append(
            {
                "episode": episode,
                "episode_sequence": sequence_index,
                "sample_count": count,
                "last_day": episode_last_days[episode],
                "terminal_observed": episode_done[episode],
                "generated_score_mean": episode_score_sums[episode] / count,
                "generated_expert_js_divergence": _js_from_histograms(
                    expert_histogram, episode_histograms[episode]
                ),
            }
        )
    episodes_frame = pd.DataFrame(episode_rows)
    episodes_frame["rolling_js_mean"] = episodes_frame[
        "generated_expert_js_divergence"
    ].rolling(rolling_episodes, min_periods=max(2, rolling_episodes // 5)).mean()

    window_rows: list[dict[str, Any]] = []
    episode_ids = episodes_frame["episode"].to_numpy(dtype=np.int64)
    for start in range(0, len(episode_ids), window_episodes):
        selected = episode_ids[start : start + window_episodes]
        pooled_histogram = np.sum(
            [episode_histograms[int(episode)] for episode in selected], axis=0
        )
        window_rows.append(
            {
                "window_index": len(window_rows) + 1,
                "episode_start": int(selected[0]),
                "episode_end": int(selected[-1]),
                "episode_count": int(selected.size),
                "sample_count": int(
                    sum(episode_counts[int(episode)] for episode in selected)
                ),
                "pooled_generated_expert_js_divergence": _js_from_histograms(
                    expert_histogram, pooled_histogram
                ),
            }
        )
    windows_frame = pd.DataFrame(window_rows)
    metadata = {
        "trajectory_path": str(trajectory_path.resolve()),
        "trajectory_size_bytes": int(trajectory_path.stat().st_size),
        "valid_scored_rows": int(total_rows),
        "dropped_nonfinite_rows": int(nonfinite_rows),
        "episode_count": int(len(episodes_frame)),
        "first_episode": int(episodes_frame["episode"].iloc[0]),
        "last_episode": int(episodes_frame["episode"].iloc[-1]),
        "episode_ids_contiguous": bool(
            np.array_equal(
                episode_ids,
                np.arange(int(episode_ids[0]), int(episode_ids[-1]) + 1),
            )
        ),
    }
    return episodes_frame, windows_frame, metadata


def _trend_summary(seed: int, windows: pd.DataFrame) -> dict[str, Any]:
    values = windows["pooled_generated_expert_js_divergence"].to_numpy(dtype=np.float64)
    indices = windows["window_index"].to_numpy(dtype=np.float64)
    edge_count = max(3, int(np.ceil(values.size * 0.20)))
    early = values[:edge_count]
    late = values[-edge_count:]
    def average_ranks(data: np.ndarray) -> list[float]:
        order = sorted(range(len(data)), key=lambda index: float(data[index]))
        ranks = [0.0] * len(order)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and data[order[end]] == data[order[start]]:
                end += 1
            average_rank = (start + 1 + end) / 2.0
            for position in range(start, end):
                ranks[order[position]] = average_rank
            start = end
        return ranks

    x_ranks = average_ranks(indices)
    y_ranks = average_ranks(values)
    x_mean = sum(x_ranks) / len(x_ranks)
    y_mean = sum(y_ranks) / len(y_ranks)
    numerator = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_ranks, y_ranks)
    )
    x_sum_squares = sum((value - x_mean) ** 2 for value in x_ranks)
    y_sum_squares = sum((value - y_mean) ** 2 for value in y_ranks)
    rho = numerator / (x_sum_squares * y_sum_squares) ** 0.5
    if len(values) > 2 and abs(rho) < 1.0:
        statistic = rho * ((len(values) - 2) / (1.0 - rho**2)) ** 0.5
        p_value = float(2.0 * student_t.sf(abs(statistic), len(values) - 2))
    else:
        p_value = 0.0 if abs(rho) == 1.0 else np.nan
    early_mean = float(early.mean())
    late_mean = float(late.mean())
    return {
        "seed": int(seed),
        "window_count": int(values.size),
        "early_window_count": int(edge_count),
        "early_js_mean": early_mean,
        "late_js_mean": late_mean,
        "late_minus_early_js": late_mean - early_mean,
        "relative_js_reduction": (
            (early_mean - late_mean) / early_mean if early_mean != 0.0 else np.nan
        ),
        "spearman_rho_episode_window_vs_js": float(rho),
        "spearman_two_sided_p_value": float(p_value),
        "descriptive_downward_trend": bool(rho < 0.0),
        "negative_trend_at_alpha_0_05": bool(rho < 0.0 and p_value < 0.05),
    }


def _load_discriminator_history(checkpoint_dir: Path, seed: int) -> pd.DataFrame:
    history_path = checkpoint_dir / "gail_validation_history.json"
    records = json.loads(history_path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(records)
    frame = frame.sort_values("training_step").drop_duplicates(
        subset=["training_step"], keep="first"
    )
    frame = frame.loc[frame["training_step"] <= 100000].copy()
    frame.insert(0, "seed", int(seed))
    frame["actor_auc_distance_from_0_5"] = (frame["actor_auc"] - 0.5).abs()
    return frame


def _plot_episode_js(seed_frames: dict[int, pd.DataFrame], output_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "SimSun", "DejaVu Serif"],
            "axes.unicode_minus": False,
            "font.size": 10,
        }
    )
    figure, axes = plt.subplots(3, 1, figsize=(9.2, 9.2), sharex=True)
    colors = {184: "#2C7FB8", 652: "#F28E2B", 187: "#2CA02C"}
    for axis, (seed, frame) in zip(axes, seed_frames.items()):
        color = colors.get(seed, "#2C7FB8")
        axis.plot(
            frame["episode"],
            frame["generated_expert_js_divergence"],
            color=color,
            alpha=0.16,
            linewidth=0.55,
            label="Per-episode JS",
        )
        axis.plot(
            frame["episode"],
            frame["rolling_js_mean"],
            color=color,
            linewidth=1.7,
            label="100-episode rolling mean",
        )
        axis.set_ylim(0.0, 0.72)
        axis.set_ylabel("JS divergence")
        axis.set_title(f"Seed {seed}", loc="left", fontsize=11)
        axis.grid(True, linestyle="--", alpha=0.32)
        axis.legend(loc="upper right", frameon=True, fontsize=8)
    axes[-1].set_xlabel("Training episode")
    figure.suptitle(
        "Historical Generated-vs-Expert Divergence under a Fixed Selected Discriminator",
        fontsize=13,
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_summary(
    window_frames: dict[int, pd.DataFrame],
    validation_frames: dict[int, pd.DataFrame],
    output_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "SimSun", "DejaVu Serif"],
            "axes.unicode_minus": False,
            "font.size": 10,
        }
    )
    colors = {184: "#2C7FB8", 652: "#F28E2B", 187: "#2CA02C"}
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    for seed, frame in window_frames.items():
        axes[0].plot(
            frame["window_index"],
            frame["pooled_generated_expert_js_divergence"],
            color=colors.get(seed),
            linewidth=1.7,
            marker="o",
            markersize=2.5,
            label=f"Seed {seed}",
        )
    axes[0].set_xlabel("100-episode window")
    axes[0].set_ylabel("Pooled JS divergence")
    axes[0].set_title("(a) Fixed-discriminator retrospective score")
    axes[0].set_ylim(bottom=0.0)
    axes[0].grid(True, linestyle="--", alpha=0.32)
    axes[0].legend(loc="best", fontsize=8)

    for seed, frame in validation_frames.items():
        axes[1].plot(
            frame["training_step"],
            frame["random_auc"],
            color=colors.get(seed),
            linewidth=1.7,
            marker="o",
            markersize=3,
            label=f"Seed {seed}",
        )
    axes[1].axhline(
        0.8,
        color="#555555",
        linestyle="--",
        linewidth=1.0,
        label="Configured AUC threshold",
    )
    axes[1].set_xlabel("TD3 parameter-update step")
    axes[1].set_ylabel("Expert-vs-random ROC-AUC")
    axes[1].set_title("(b) Online discriminator validation history")
    axes[1].set_ylim(0.5, 1.01)
    axes[1].grid(True, linestyle="--", alpha=0.32)
    axes[1].legend(loc="lower left", fontsize=8)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def evaluate_seed(
    seed: int,
    *,
    checkpoint_root: Path,
    trajectory_root: Path,
    expert_csv: Path,
    output_dir: Path,
    device: torch.device,
    bins: int,
    chunk_size: int,
    score_batch_size: int,
    rolling_episodes: int,
    window_episodes: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    checkpoint_dir = checkpoint_root / f"seed_{seed}"
    trajectory_path = trajectory_root / f"seed_{seed}" / "production1_daily_trajectory.csv"
    discriminator_path = checkpoint_dir / "discriminator_best_validation.pth"
    rms_path = checkpoint_dir / "obs_rms_params.pth"
    split_manifest_path = checkpoint_dir / "expert_split.json"
    required = [trajectory_path, discriminator_path, rms_path, split_manifest_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Seed {seed} is missing required files: {missing}")

    discriminator = FinalDiscriminator().to(device)
    discriminator.load_state_dict(_load_state_dict(discriminator_path, device))
    rms = _load_rms(rms_path, device)
    _, expert_histogram, expert_metadata = _load_expert_reference(
        expert_csv,
        split_manifest_path,
        discriminator,
        rms,
        device,
        score_batch_size,
        bins,
    )
    print(f"Seed {seed}: replaying {trajectory_path}")
    episodes, windows, trajectory_metadata = _read_and_score_trajectory(
        trajectory_path,
        discriminator,
        rms,
        expert_histogram,
        device,
        bins=bins,
        chunk_size=chunk_size,
        score_batch_size=score_batch_size,
        rolling_episodes=rolling_episodes,
        window_episodes=window_episodes,
    )
    episodes.insert(0, "seed", int(seed))
    windows.insert(0, "seed", int(seed))
    validation = _load_discriminator_history(checkpoint_dir, seed)
    episodes.to_csv(output_dir / f"seed_{seed}_per_episode_js.csv", index=False)
    windows.to_csv(output_dir / f"seed_{seed}_per_{window_episodes}_episode_js.csv", index=False)
    validation.to_csv(output_dir / f"seed_{seed}_discriminator_validation_history.csv", index=False)
    metadata = {
        "seed": int(seed),
        "fixed_discriminator": str(discriminator_path.resolve()),
        "fixed_discriminator_sha256": file_sha256(discriminator_path),
        "normalization_parameters": str(rms_path.resolve()),
        "normalization_parameters_sha256": file_sha256(rms_path),
        "expert_reference": expert_metadata,
        "trajectory": trajectory_metadata,
    }
    return episodes, windows, validation, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--trajectory-root", type=Path, default=DEFAULT_TRAJECTORY_ROOT)
    parser.add_argument("--expert-csv", type=Path, default=DEFAULT_EXPERT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=100000)
    parser.add_argument("--score-batch-size", type=int, default=32768)
    parser.add_argument("--rolling-episodes", type=int, default=100)
    parser.add_argument("--window-episodes", type=int, default=100)
    args = parser.parse_args()

    if args.bins < 2:
        raise ValueError("--bins must be at least 2.")
    if args.rolling_episodes < 2 or args.window_episodes < 2:
        raise ValueError("Rolling and pooled window sizes must be at least 2 episodes.")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase-2 retrospective evaluation device: {device}")

    episode_frames: dict[int, pd.DataFrame] = {}
    window_frames: dict[int, pd.DataFrame] = {}
    validation_frames: dict[int, pd.DataFrame] = {}
    run_metadata: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        episodes, windows, validation, metadata = evaluate_seed(
            seed,
            checkpoint_root=args.checkpoint_root.resolve(),
            trajectory_root=args.trajectory_root.resolve(),
            expert_csv=args.expert_csv.resolve(),
            output_dir=output_dir,
            device=device,
            bins=args.bins,
            chunk_size=args.chunk_size,
            score_batch_size=args.score_batch_size,
            rolling_episodes=args.rolling_episodes,
            window_episodes=args.window_episodes,
        )
        episode_frames[int(seed)] = episodes
        window_frames[int(seed)] = windows
        validation_frames[int(seed)] = validation
        run_metadata.append(metadata)
        trend = _trend_summary(seed, windows)
        trend.update(
            {
                "discriminator_validation_first_step": int(
                    validation["training_step"].iloc[0]
                ),
                "discriminator_validation_last_step": int(
                    validation["training_step"].iloc[-1]
                ),
                "random_auc_first": float(validation["random_auc"].iloc[0]),
                "random_auc_last": float(validation["random_auc"].iloc[-1]),
                "random_auc_min": float(validation["random_auc"].min()),
                "random_auc_all_at_least_0_8": bool(
                    (validation["random_auc"] >= 0.8).all()
                ),
            }
        )
        summary_rows.append(trend)

    all_episodes = pd.concat(episode_frames.values(), ignore_index=True)
    all_windows = pd.concat(window_frames.values(), ignore_index=True)
    all_validation = pd.concat(validation_frames.values(), ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    all_episodes.to_csv(output_dir / "three_seed_per_episode_js.csv", index=False)
    all_windows.to_csv(
        output_dir / f"three_seed_per_{args.window_episodes}_episode_js.csv",
        index=False,
    )
    all_validation.to_csv(
        output_dir / "three_seed_discriminator_validation_history.csv", index=False
    )
    summary.to_csv(output_dir / "three_seed_historical_evolution_summary.csv", index=False)

    methodology = {
        "analysis": "phase_2_retrospective_historical_gail_evolution",
        "seeds": [int(seed) for seed in args.seeds],
        "device": str(device),
        "fixed_ruler": (
            "Each seed uses discriminator_best_validation.pth, the same selected "
            "discriminator evaluated in phase 1."
        ),
        "expert_reference": (
            "Complete expert episodes in the held-out final-test partition are scored "
            "by the corresponding fixed discriminator."
        ),
        "historical_generated_sample": (
            "Logged production1 state-action pairs. Actions are behavior actions "
            "actually applied during training and include TD3 exploration noise."
        ),
        "normalization": (
            "States and actions use the seed checkpoint's obs_rms_params.pth; states "
            "are clipped to [-5, 5] after normalization."
        ),
        "js_definition": (
            "Squared scipy.spatial.distance.jensenshannon over 50 fixed bins on "
            "discriminator sigmoid scores in [0, 1]."
        ),
        "per_episode_result": (
            "One JS divergence is computed from each episode's score histogram and "
            "the fixed held-out expert score histogram."
        ),
        "pooled_window_result": (
            f"Histograms from each consecutive {args.window_episodes} episodes are "
            "pooled before JS calculation to reduce short-episode sampling noise."
        ),
        "trend_test": (
            "Two-sided Spearman correlation between pooled-window index and JS. A "
            "negative coefficient denotes a descriptive downward trend; alpha=0.05 "
            "is used only for the reported correlation test."
        ),
        "discriminator_history": (
            "The saved online validation history is deduplicated by TD3 update step "
            "and restricted to steps <=100000, after which parameter updates stopped."
        ),
        "interpretation_limit": (
            "Fixed-discriminator historical JS measures evolution of the logged behavior "
            "distribution under a common ruler. It cannot by itself identify discriminator "
            "parameter evolution or deterministic Actor-only evolution."
        ),
        "configuration": {
            "histogram_bins": int(args.bins),
            "rolling_episode_count": int(args.rolling_episodes),
            "pooled_window_episode_count": int(args.window_episodes),
            "trajectory_chunk_size": int(args.chunk_size),
            "score_batch_size": int(args.score_batch_size),
        },
        "inputs": run_metadata,
    }
    (output_dir / "methodology_and_input_manifest.json").write_text(
        json.dumps(_json_safe(methodology), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plotter_path = PROJECT_DIR / "plot_historical_gail_evolution.py"
    plot_python = Path(sys.executable)
    executable_parts = [part.lower() for part in plot_python.parts]
    if "envs" in executable_parts and len(plot_python.parents) >= 3:
        conda_base_python = plot_python.parents[2] / "python.exe"
        if conda_base_python.exists():
            plot_python = conda_base_python
    plot_command = [
        str(plot_python),
        str(plotter_path),
        "--output-dir",
        str(output_dir),
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--window-episodes",
        str(args.window_episodes),
    ]
    subprocess.run(plot_command, check=True)
    print(f"Phase-2 outputs written to: {output_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
