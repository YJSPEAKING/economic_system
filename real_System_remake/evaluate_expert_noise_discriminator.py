"""Evaluate whether fixed final discriminators separate expert actions from noisy expert actions.

The evaluation uses complete expert episodes from the held-out final-test split.
For every held-out expert state, independent zero-mean Gaussian noise is added
to the recorded expert action and the result is clipped to the legal action
range. The recorded and perturbed actions are therefore paired on the same
state and expert episode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from real_System_remake.final_gail_statistical_eval import (
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_EXPERT_CSV,
    DEFAULT_SEEDS,
    FinalDiscriminator,
    _episode_means,
    _holm_adjust,
    _load_held_out_expert_data,
    _load_rms,
    _load_state_dict,
    _paired_permutation_greater,
    _sha256,
    compare_distributions,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = (
    PROJECT_DIR / "analysis_plots" / "expert_vs_noisy_expert_discriminator_eval"
)


def _histogram_overlap(
    expert_scores: np.ndarray, target_scores: np.ndarray, bins: int
) -> float:
    """Return the overlap coefficient of normalized score histograms."""
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    expert_hist, _ = np.histogram(expert_scores, bins=bin_edges)
    target_hist, _ = np.histogram(target_scores, bins=bin_edges)
    expert_prob = expert_hist.astype(np.float64)
    target_prob = target_hist.astype(np.float64)
    expert_prob /= max(expert_prob.sum(), 1.0)
    target_prob /= max(target_prob.sum(), 1.0)
    return float(np.minimum(expert_prob, target_prob).sum())


def _score_expert_and_noisy_actions(
    discriminator: FinalDiscriminator,
    states_raw: np.ndarray,
    expert_actions_raw: np.ndarray,
    noisy_actions_raw: np.ndarray,
    rms: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    expert_scores: list[np.ndarray] = []
    noisy_scores: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(states_raw), batch_size):
            end = min(start + batch_size, len(states_raw))
            states = torch.as_tensor(
                states_raw[start:end], dtype=torch.float32, device=device
            )
            expert_actions = torch.as_tensor(
                expert_actions_raw[start:end], dtype=torch.float32, device=device
            )
            noisy_actions = torch.as_tensor(
                noisy_actions_raw[start:end], dtype=torch.float32, device=device
            )
            states_norm = torch.clamp(
                (states - rms["mean"]) / torch.sqrt(rms["var"] + 1e-8),
                -5.0,
                5.0,
            )
            action_scale = torch.sqrt(rms["act_var"] + 1e-8)
            expert_actions_norm = (expert_actions - rms["act_mean"]) / action_scale
            noisy_actions_norm = (noisy_actions - rms["act_mean"]) / action_scale
            expert_scores.append(
                torch.sigmoid(discriminator(states_norm, expert_actions_norm))
                .cpu()
                .numpy()
            )
            noisy_scores.append(
                torch.sigmoid(discriminator(states_norm, noisy_actions_norm))
                .cpu()
                .numpy()
            )
    return (
        np.concatenate(expert_scores, axis=0).reshape(-1),
        np.concatenate(noisy_scores, axis=0).reshape(-1),
    )


def _save_sample_table(
    output_path: Path,
    source_rows: np.ndarray,
    episode_labels: np.ndarray,
    states: np.ndarray,
    expert_actions: np.ndarray,
    sampled_noise: np.ndarray,
    noisy_actions: np.ndarray,
    expert_scores: np.ndarray,
    noisy_scores: np.ndarray,
) -> None:
    columns: dict[str, Any] = {
        "source_row": source_rows,
        "expert_episode_id": episode_labels,
    }
    for index in range(states.shape[1]):
        columns[f"state_{index}"] = states[:, index]
    applied_noise = noisy_actions - expert_actions
    for index in range(expert_actions.shape[1]):
        columns[f"expert_action_{index}"] = expert_actions[:, index]
        columns[f"sampled_noise_{index}"] = sampled_noise[:, index]
        columns[f"noisy_expert_action_{index}"] = noisy_actions[:, index]
        columns[f"applied_noise_after_clipping_{index}"] = applied_noise[:, index]
    columns["expert_score"] = expert_scores
    columns["noisy_expert_score"] = noisy_scores
    pd.DataFrame(columns).to_csv(output_path, index=False, encoding="utf-8-sig")


def _plot_seed(
    seed: int,
    expert_scores: np.ndarray,
    noisy_scores: np.ndarray,
    metrics: dict[str, Any],
    output_path: Path,
) -> None:
    comparison = metrics["noisy_expert_vs_expert"]
    noise_std = metrics["noise_protocol"]["noise_std_raw_action_units"]
    figure, axis = plt.subplots(figsize=(10.5, 6.2))
    axis.hist(
        expert_scores,
        bins=50,
        range=(0.0, 1.0),
        alpha=0.52,
        color="#2ca02c",
        label=f"Expert (Mean={metrics['expert']['mean']:.3f})",
    )
    axis.hist(
        noisy_scores,
        bins=50,
        range=(0.0, 1.0),
        alpha=0.42,
        color="#9467bd",
        label=(
            f"Noisy expert (Mean={comparison['target_mean']:.3f}, "
            f"JS={comparison['js_divergence']:.3f}, "
            f"Overlap={comparison['histogram_overlap_coefficient']:.3f}, "
            f"AUC={comparison['roc_auc_expert_positive']:.3f}, "
            f"P_diff={comparison['paired_permutation_p_value']:.2e})"
        ),
    )
    axis.axvline(
        metrics["expert"]["mean"], color="#1b6e1b", linestyle="--", linewidth=1.5
    )
    axis.axvline(
        comparison["target_mean"], color="#5e3c99", linestyle="--", linewidth=1.5
    )
    axis.set_title(
        f"Expert vs Noisy-Expert Discriminator Scores "
        f"(seed={seed}, noise std={noise_std:.2f})"
    )
    axis.set_xlabel("Discriminator confidence score")
    axis.set_ylabel("Sample count")
    axis.set_xlim(0.0, 1.0)
    axis.grid(axis="y", linestyle="--", alpha=0.25)
    axis.legend(fontsize=9, frameon=True, framealpha=0.95)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def evaluate_seed(
    seed: int,
    *,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    expert_csv: Path = DEFAULT_EXPERT_CSV,
    noise_std: float = 0.10,
    action_bound: float = 0.5,
    bins: int = 100,
    bootstrap_iterations: int = 500,
    permutation_iterations: int = 5000,
    inference_batch_size: int = 4096,
    metric_random_seed: int = 20260731,
    device_name: str | None = None,
) -> dict[str, Any]:
    if noise_std <= 0.0:
        raise ValueError("noise_std must be positive.")
    checkpoint_dir = Path(checkpoint_root) / f"seed_{seed}"
    output_dir = Path(output_root) / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    discriminator_path = checkpoint_dir / "discriminator_best_validation.pth"
    rms_path = checkpoint_dir / "obs_rms_params.pth"
    split_manifest_path = checkpoint_dir / "expert_split.json"
    required = (discriminator_path, rms_path, split_manifest_path, Path(expert_csv))
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing evaluation inputs for seed {seed}: {missing}")

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    discriminator = FinalDiscriminator(s_dim=33, a_dim=4).to(device)
    discriminator.load_state_dict(_load_state_dict(discriminator_path, device))
    discriminator.eval()
    rms = _load_rms(rms_path, device)
    source_rows, episode_labels, evaluation_data, split_metadata = (
        _load_held_out_expert_data(Path(expert_csv), split_manifest_path)
    )
    states_raw = evaluation_data[:, :33]
    expert_actions_raw = evaluation_data[:, 33:37]

    seed_sequence = np.random.SeedSequence(metric_random_seed + int(seed))
    noise_seed, bootstrap_seed, permutation_seed = seed_sequence.spawn(3)
    noise_rng = np.random.default_rng(noise_seed)
    bootstrap_rng = np.random.default_rng(bootstrap_seed)
    permutation_rng = np.random.default_rng(permutation_seed)
    sampled_noise = noise_rng.normal(
        loc=0.0, scale=noise_std, size=expert_actions_raw.shape
    ).astype(np.float32)
    unbounded_actions = expert_actions_raw + sampled_noise
    noisy_actions_raw = np.clip(
        unbounded_actions, -action_bound, action_bound
    ).astype(np.float32)
    clipped_elements = (unbounded_actions < -action_bound) | (
        unbounded_actions > action_bound
    )

    expert_scores, noisy_scores = _score_expert_and_noisy_actions(
        discriminator,
        states_raw,
        expert_actions_raw,
        noisy_actions_raw,
        rms,
        device,
        inference_batch_size,
    )
    comparison = compare_distributions(
        expert_scores,
        noisy_scores,
        episode_labels,
        bins=bins,
        bootstrap_iterations=bootstrap_iterations,
        auc_equivalence_margin=0.05,
        rng=bootstrap_rng,
    )
    comparison["histogram_overlap_coefficient"] = _histogram_overlap(
        expert_scores, noisy_scores, bins
    )
    comparison["histogram_overlap_below_0_5"] = bool(
        comparison["histogram_overlap_coefficient"] < 0.5
    )
    expert_episode_means = _episode_means(expert_scores, episode_labels)
    noisy_episode_means = _episode_means(noisy_scores, episode_labels)
    comparison.update(
        _paired_permutation_greater(
            expert_episode_means,
            noisy_episode_means,
            permutation_iterations,
            permutation_rng,
        )
    )
    comparison["distinguishable_by_prespecified_score_criteria"] = bool(
        comparison["expert_score_greater_at_alpha_0_05"]
        and comparison["roc_auc_ci95_low"] > 0.5
    )

    applied_noise = noisy_actions_raw - expert_actions_raw
    metrics: dict[str, Any] = {
        "seed": int(seed),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "expert_csv": str(Path(expert_csv).resolve()),
        "evaluation_partition": "final_test_complete_expert_episodes",
        "statistical_unit_for_inference": "final_test_expert_episode",
        "score_pairing": "same_final_test_expert_state_and_recorded_action_before_noise",
        "noise_protocol": {
            "distribution": "independent_zero_mean_gaussian_per_action_component",
            "noise_std_raw_action_units": float(noise_std),
            "legal_action_interval": [-float(action_bound), float(action_bound)],
            "post_noise_operation": "clip_to_legal_action_interval",
            "metric_random_seed": int(metric_random_seed + int(seed)),
            "element_clipping_fraction": float(clipped_elements.mean()),
            "row_clipping_fraction": float(clipped_elements.any(axis=1).mean()),
            "applied_noise_mean_absolute": float(np.mean(np.abs(applied_noise))),
            "applied_noise_rmse": float(np.sqrt(np.mean(np.square(applied_noise)))),
        },
        "histogram_bins_for_js": int(bins),
        "primary_test": (
            "one-sided paired sign-flip permutation test on held-out episode-mean "
            "discriminator score difference"
        ),
        "prespecified_separation_criteria": (
            "Holm-adjusted paired permutation p < 0.05 across seeds and the "
            "cluster-bootstrap 95% lower confidence bound for expert-positive "
            "ROC-AUC > 0.5"
        ),
        "expert_split": split_metadata,
        "expert": {
            "count": int(expert_scores.size),
            "episode_count": int(np.unique(episode_labels).size),
            "mean": float(np.mean(expert_scores)),
            "std": float(np.std(expert_scores, ddof=1)),
        },
        "noisy_expert_vs_expert": comparison,
        "checkpoint_sha256": {
            "discriminator_best_validation.pth": _sha256(discriminator_path),
            "obs_rms_params.pth": _sha256(rms_path),
            "expert_split.json": _sha256(split_manifest_path),
        },
    }

    _save_sample_table(
        output_dir / "evaluation_samples_and_scores.csv",
        source_rows,
        episode_labels,
        states_raw,
        expert_actions_raw,
        sampled_noise,
        noisy_actions_raw,
        expert_scores,
        noisy_scores,
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, ensure_ascii=False, indent=2)
    flat_metrics = {
        "seed": int(seed),
        "expert_count": metrics["expert"]["count"],
        "expert_episode_count": metrics["expert"]["episode_count"],
        "noise_std": float(noise_std),
        "element_clipping_fraction": metrics["noise_protocol"][
            "element_clipping_fraction"
        ],
        "row_clipping_fraction": metrics["noise_protocol"]["row_clipping_fraction"],
        "expert_mean": metrics["expert"]["mean"],
        "noisy_expert_mean": comparison["target_mean"],
        "js_divergence": comparison["js_divergence"],
        "js_ci95_low": comparison["js_divergence_ci95_low"],
        "js_ci95_high": comparison["js_divergence_ci95_high"],
        "histogram_overlap_coefficient": comparison[
            "histogram_overlap_coefficient"
        ],
        "histogram_overlap_below_0_5": comparison[
            "histogram_overlap_below_0_5"
        ],
        "roc_auc": comparison["roc_auc_expert_positive"],
        "roc_auc_ci95_low": comparison["roc_auc_ci95_low"],
        "roc_auc_ci95_high": comparison["roc_auc_ci95_high"],
        "paired_mean_score_difference": comparison["paired_mean_score_difference"],
        "paired_permutation_p_value": comparison["paired_permutation_p_value"],
        "expert_score_greater": comparison["expert_score_greater_at_alpha_0_05"],
        "distinguishable_score_criteria": comparison[
            "distinguishable_by_prespecified_score_criteria"
        ],
        "wasserstein_distance": comparison["wasserstein_distance"],
    }
    pd.DataFrame([flat_metrics]).to_csv(
        output_dir / "metrics.csv", index=False, encoding="utf-8-sig"
    )
    _plot_seed(
        seed,
        expert_scores,
        noisy_scores,
        metrics,
        output_dir / "expert_vs_noisy_expert_score_distributions.png",
    )
    return metrics


def write_summary(output_root: Path, seeds: tuple[int, ...]) -> Path:
    rows: list[dict[str, Any]] = []
    score_tables: dict[int, pd.DataFrame] = {}
    for seed in seeds:
        seed_dir = Path(output_root) / f"seed_{seed}"
        rows.append(pd.read_csv(seed_dir / "metrics.csv").iloc[0].to_dict())
        score_tables[int(seed)] = pd.read_csv(
            seed_dir / "evaluation_samples_and_scores.csv"
        )
    frame = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    frame["paired_permutation_p_value_holm"] = _holm_adjust(
        frame["paired_permutation_p_value"].to_numpy(dtype=np.float64)
    )
    frame["distinguishable_score_criteria_holm"] = (
        (frame["paired_permutation_p_value_holm"] < 0.05)
        & (frame["roc_auc_ci95_low"] > 0.5)
    )
    summary_path = Path(output_root) / "three_seed_expert_noise_summary.csv"
    frame.to_csv(summary_path, index=False, encoding="utf-8-sig")

    figure, axes = plt.subplots(1, len(seeds), figsize=(6.5 * len(seeds), 5.5))
    if len(seeds) == 1:
        axes = [axes]
    for axis, seed in zip(axes, seeds):
        table = score_tables[int(seed)]
        row = frame.loc[frame["seed"].astype(int) == int(seed)].iloc[0]
        axis.hist(
            table["expert_score"],
            bins=50,
            range=(0, 1),
            alpha=0.52,
            color="#2ca02c",
            label=f"Expert (Mean={row['expert_mean']:.3f})",
        )
        axis.hist(
            table["noisy_expert_score"],
            bins=50,
            range=(0, 1),
            alpha=0.42,
            color="#9467bd",
            label=(
                f"Noisy expert (Mean={row['noisy_expert_mean']:.3f}, "
                f"JS={row['js_divergence']:.3f}, "
                f"Overlap={row['histogram_overlap_coefficient']:.3f}, "
                f"AUC={row['roc_auc']:.3f}, "
                f"P={row['paired_permutation_p_value']:.2e})"
            ),
        )
        axis.set_title(
            f"seed={int(seed)}, noise std={float(row['noise_std']):.2f}"
        )
        axis.set_xlabel("Discriminator confidence score")
        axis.set_xlim(0.0, 1.0)
        axis.grid(axis="y", linestyle="--", alpha=0.25)
        axis.legend(fontsize=8, framealpha=0.95)
    axes[0].set_ylabel("Sample count")
    figure.suptitle("Expert vs Noisy-Expert Discriminator Evaluation", y=1.02)
    figure.tight_layout()
    figure.savefig(
        Path(output_root) / "three_seed_expert_vs_noisy_expert.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    conclusion = {
        "all_seeds_support_expert_noisy_expert_separation": bool(
            frame["distinguishable_score_criteria_holm"].all()
        ),
        "criterion": (
            "For every seed: Holm-adjusted paired permutation p < 0.05 and "
            "cluster-bootstrap 95% ROC-AUC lower bound > 0.5."
        ),
        "seed_results": {
            str(int(row.seed)): bool(row.distinguishable_score_criteria_holm)
            for row in frame.itertuples(index=False)
        },
        "all_seeds_histogram_overlap_below_0_5": bool(
            frame["histogram_overlap_below_0_5"].astype(bool).all()
        ),
        "histogram_overlap_definition": (
            "Sum of the binwise minima of two normalized discriminator-score "
            "histograms using the same 100 bins on [0, 1]."
        ),
    }
    with (Path(output_root) / "conclusion_gate.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(conclusion, stream, ensure_ascii=False, indent=2)
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--expert-csv", type=Path, default=DEFAULT_EXPERT_CSV)
    parser.add_argument("--noise-std", type=float, default=0.10)
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    parser.add_argument("--permutation-iterations", type=int, default=5000)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(args.seeds)
    args.output_root.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        metrics = evaluate_seed(
            seed,
            checkpoint_root=args.checkpoint_root,
            output_root=args.output_root,
            expert_csv=args.expert_csv,
            noise_std=args.noise_std,
            bootstrap_iterations=args.bootstrap_iterations,
            permutation_iterations=args.permutation_iterations,
            device_name=args.device,
        )
        comparison = metrics["noisy_expert_vs_expert"]
        print(
            f"seed={seed}: expert mean={metrics['expert']['mean']:.4f}, "
            f"noisy mean={comparison['target_mean']:.4f}, "
            f"JS={comparison['js_divergence']:.4f}, "
            f"AUC={comparison['roc_auc_expert_positive']:.4f}, "
            f"P_diff={comparison['paired_permutation_p_value']:.4e}"
        )
    summary_path = write_summary(args.output_root, seeds)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
