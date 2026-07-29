"""Evaluate final online-GAIL Actor and discriminator checkpoints.

The evaluator uses only held-out expert episodes.  For every held-out expert
state it compares the discriminator score assigned to the recorded expert
action, the deterministic production Actor action, and a uniform-random
action.  All paired inputs and scores are saved for reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.stats import ks_2samp, rankdata, t as student_t, wasserstein_distance
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from real_System_remake.expert_data_split import load_expert_episode_split
except ModuleNotFoundError:
    from expert_data_split import load_expert_episode_split


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_ROOT = PROJECT_DIR / "checkpoints" / "final_weights" / "GAIL+TD3"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "analysis_plots" / "final_gail_discriminator_eval"
DEFAULT_EXPERT_CSV = PROJECT_DIR / "expert_data_production1_collected.csv"
DEFAULT_SEEDS = (184, 652, 187)


class FinalActor(nn.Module):
    """Production-enterprise Actor architecture saved by System.py."""

    def __init__(self, s_dim: int = 33, a_dim: int = 4, a_bound: float = 0.5):
        super().__init__()
        self.l1 = nn.Linear(s_dim, 128)
        self.l2 = nn.Linear(128, 32)
        self.l3 = nn.Linear(32, a_dim)
        self.a_bound = a_bound

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        action = torch.tanh(self.l1(state))
        action = F.leaky_relu(self.l2(action))
        return self.a_bound * torch.tanh(self.l3(action))


class FinalDiscriminator(nn.Module):
    """Online-GAIL discriminator architecture."""

    def __init__(self, s_dim: int = 33, a_dim: int = 4, hidden_size: int = 100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + a_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


def _load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_rms(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    required = ("mean", "var", "act_mean", "act_var")
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"Missing normalization fields in {path}: {missing}")
    return {
        key: torch.as_tensor(payload[key], dtype=torch.float32, device=device)
        for key in required
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _paired_finite(
    expert_scores: np.ndarray, target_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    expert = np.asarray(expert_scores, dtype=np.float64).reshape(-1)
    target = np.asarray(target_scores, dtype=np.float64).reshape(-1)
    if expert.shape != target.shape:
        raise ValueError("Paired discriminator score arrays must have the same shape.")
    mask = np.isfinite(expert) & np.isfinite(target)
    if not mask.any():
        raise ValueError("No finite paired discriminator scores were found.")
    return expert[mask], target[mask]


def _js_divergence(expert: np.ndarray, target: np.ndarray, bins: int) -> float:
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    expert_hist, _ = np.histogram(expert, bins=bin_edges)
    target_hist, _ = np.histogram(target, bins=bin_edges)
    expert_prob = expert_hist.astype(np.float64)
    target_prob = target_hist.astype(np.float64)
    expert_prob /= max(expert_prob.sum(), 1.0)
    target_prob /= max(target_prob.sum(), 1.0)
    return float(distance.jensenshannon(expert_prob, target_prob) ** 2)


def _roc_auc(expert_positive: np.ndarray, target_negative: np.ndarray) -> float:
    positive = np.asarray(expert_positive, dtype=np.float64).reshape(-1)
    negative = np.asarray(target_negative, dtype=np.float64).reshape(-1)
    if positive.size == 0 or negative.size == 0:
        raise ValueError("ROC-AUC requires non-empty positive and negative samples.")
    values = np.concatenate((negative, positive))
    ranks = rankdata(values, method="average")
    positive_rank_sum = float(ranks[negative.size :].sum())
    numerator = positive_rank_sum - positive.size * (positive.size + 1) / 2.0
    return float(numerator / (positive.size * negative.size))


def _episode_means(values: np.ndarray, episode_labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(episode_labels, dtype=np.int64).reshape(-1)
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if labels.shape != data.shape:
        raise ValueError("Episode labels and score values must have the same length.")
    return np.asarray(
        [data[labels == episode_id].mean() for episode_id in np.unique(labels)],
        dtype=np.float64,
    )


def _paired_tost(
    expert_episode_means: np.ndarray,
    target_episode_means: np.ndarray,
    equivalence_margin: float,
) -> dict[str, float | int | bool]:
    """Paired TOST for equivalence of episode-mean discriminator scores."""
    differences = np.asarray(expert_episode_means) - np.asarray(target_episode_means)
    sample_count = int(differences.size)
    if sample_count < 2:
        raise ValueError("Paired TOST requires at least two held-out episodes.")
    mean_difference = float(differences.mean())
    standard_error = float(differences.std(ddof=1) / np.sqrt(sample_count))
    degrees_of_freedom = sample_count - 1
    if standard_error == 0.0:
        p_lower = 0.0 if mean_difference > -equivalence_margin else 1.0
        p_upper = 0.0 if mean_difference < equivalence_margin else 1.0
        ci90_low = mean_difference
        ci90_high = mean_difference
    else:
        lower_statistic = (mean_difference + equivalence_margin) / standard_error
        upper_statistic = (mean_difference - equivalence_margin) / standard_error
        p_lower = float(student_t.sf(lower_statistic, degrees_of_freedom))
        p_upper = float(student_t.cdf(upper_statistic, degrees_of_freedom))
        critical = float(student_t.ppf(0.95, degrees_of_freedom))
        ci90_low = mean_difference - critical * standard_error
        ci90_high = mean_difference + critical * standard_error
    p_value = max(p_lower, p_upper)
    return {
        "equivalence_episode_count": sample_count,
        "paired_mean_score_difference": mean_difference,
        "paired_mean_score_difference_ci90_low": float(ci90_low),
        "paired_mean_score_difference_ci90_high": float(ci90_high),
        "mean_equivalence_margin": float(equivalence_margin),
        "mean_equivalence_p_lower": float(p_lower),
        "mean_equivalence_p_upper": float(p_upper),
        "mean_equivalence_p_value": float(p_value),
        "mean_equivalent_at_alpha_0_05": bool(p_value < 0.05),
    }


def _paired_permutation_greater(
    expert_episode_means: np.ndarray,
    target_episode_means: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
) -> dict[str, float | int | bool]:
    """One-sided paired sign-flip test for expert score greater than target."""
    differences = np.asarray(expert_episode_means) - np.asarray(target_episode_means)
    observed = float(differences.mean())
    extreme_count = 0
    batch_size = 256
    completed = 0
    while completed < iterations:
        current = min(batch_size, iterations - completed)
        signs = rng.choice((-1.0, 1.0), size=(current, differences.size))
        null_means = (signs * differences).mean(axis=1)
        extreme_count += int(np.count_nonzero(null_means >= observed))
        completed += current
    p_value = float((extreme_count + 1) / (iterations + 1))
    return {
        "permutation_episode_count": int(differences.size),
        "paired_mean_score_difference": observed,
        "paired_permutation_iterations": int(iterations),
        "paired_permutation_alternative": "expert_score_greater_than_target_score",
        "paired_permutation_p_value": p_value,
        "expert_score_greater_at_alpha_0_05": bool(p_value < 0.05),
    }


def _cluster_bootstrap(
    expert: np.ndarray,
    target: np.ndarray,
    episode_labels: np.ndarray,
    *,
    bins: int,
    iterations: int,
    auc_equivalence_margin: float,
    rng: np.random.Generator,
) -> dict[str, float | int | bool]:
    """Bootstrap complete held-out episodes with replacement."""
    unique_episodes = np.unique(episode_labels)
    episode_rows = [np.flatnonzero(episode_labels == value) for value in unique_episodes]
    mean_differences = np.empty(iterations, dtype=np.float64)
    js_values = np.empty(iterations, dtype=np.float64)
    auc_values = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        selected = rng.integers(0, len(episode_rows), size=len(episode_rows))
        row_indices = np.concatenate([episode_rows[item] for item in selected])
        expert_sample = expert[row_indices]
        target_sample = target[row_indices]
        mean_differences[index] = np.mean(expert_sample - target_sample)
        js_values[index] = _js_divergence(expert_sample, target_sample, bins)
        auc_values[index] = _roc_auc(expert_sample, target_sample)

    def interval(values: np.ndarray, lower: float, upper: float) -> tuple[float, float]:
        bounds = np.percentile(values, [lower, upper])
        return float(bounds[0]), float(bounds[1])

    mean_ci95 = interval(mean_differences, 2.5, 97.5)
    js_ci95 = interval(js_values, 2.5, 97.5)
    auc_ci90 = interval(auc_values, 5.0, 95.0)
    auc_ci95 = interval(auc_values, 2.5, 97.5)
    auc_lower = 0.5 - auc_equivalence_margin
    auc_upper = 0.5 + auc_equivalence_margin
    return {
        "cluster_bootstrap_iterations": int(iterations),
        "cluster_bootstrap_unit": "held_out_expert_episode",
        "paired_mean_score_difference_ci95_low": mean_ci95[0],
        "paired_mean_score_difference_ci95_high": mean_ci95[1],
        "js_divergence_ci95_low": js_ci95[0],
        "js_divergence_ci95_high": js_ci95[1],
        "roc_auc_ci90_low": auc_ci90[0],
        "roc_auc_ci90_high": auc_ci90[1],
        "roc_auc_ci95_low": auc_ci95[0],
        "roc_auc_ci95_high": auc_ci95[1],
        "roc_auc_equivalence_margin": float(auc_equivalence_margin),
        "roc_auc_equivalent_ci90": bool(
            auc_ci90[0] >= auc_lower and auc_ci90[1] <= auc_upper
        ),
    }


def compare_distributions(
    expert_scores: np.ndarray,
    target_scores: np.ndarray,
    episode_labels: np.ndarray,
    *,
    bins: int,
    bootstrap_iterations: int,
    auc_equivalence_margin: float,
    rng: np.random.Generator,
) -> dict[str, float | int | bool | str]:
    """Compute paired descriptive distances and cluster-bootstrap intervals."""
    expert, target = _paired_finite(expert_scores, target_scores)
    labels = np.asarray(episode_labels, dtype=np.int64).reshape(-1)
    if labels.size != expert.size:
        raise ValueError("Episode labels do not match discriminator score rows.")
    full_ks = ks_2samp(expert, target, method="auto")
    metrics: dict[str, float | int | bool | str] = {
        "target_count": int(target.size),
        "target_mean": float(np.mean(target)),
        "target_std": float(np.std(target, ddof=1)) if target.size > 1 else 0.0,
        "js_divergence": _js_divergence(expert, target, bins),
        "roc_auc_expert_positive": _roc_auc(expert, target),
        "wasserstein_distance": float(wasserstein_distance(expert, target)),
        "ks_full_statistic_descriptive": float(full_ks.statistic),
        "ks_full_p_value_descriptive": float(full_ks.pvalue),
    }
    metrics.update(
        _cluster_bootstrap(
            expert,
            target,
            labels,
            bins=bins,
            iterations=bootstrap_iterations,
            auc_equivalence_margin=auc_equivalence_margin,
            rng=rng,
        )
    )
    return metrics


def _load_held_out_expert_data(
    expert_csv: Path, split_manifest_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    with split_manifest_path.open("r", encoding="utf-8") as stream:
        saved_metadata = json.load(stream)
    split = load_expert_episode_split(expert_csv)
    checked_fields = (
        "expert_csv_sha256",
        "split_unit",
        "split_seed",
        "test_fraction_requested",
        "total_episodes",
        "train_episodes",
        "test_episodes",
        "train_rows",
        "test_rows",
        "test_episode_ids",
    )
    mismatches = [
        field
        for field in checked_fields
        if saved_metadata.get(field) != split.metadata.get(field)
    ]
    if mismatches:
        raise RuntimeError(
            "The checkpoint expert split does not match the current expert data: "
            + ", ".join(mismatches)
        )
    return (
        split.test_source_rows,
        split.test_episode_labels,
        split.test_values,
        split.metadata,
    )


def _score_actions(
    actor: FinalActor,
    discriminator: FinalDiscriminator,
    states_raw: np.ndarray,
    expert_actions_raw: np.ndarray,
    random_actions_raw: np.ndarray,
    rms: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    actor_actions: list[np.ndarray] = []
    expert_scores: list[np.ndarray] = []
    actor_scores: list[np.ndarray] = []
    random_scores: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(states_raw), batch_size):
            end = min(start + batch_size, len(states_raw))
            states = torch.as_tensor(states_raw[start:end], dtype=torch.float32, device=device)
            expert_actions = torch.as_tensor(
                expert_actions_raw[start:end], dtype=torch.float32, device=device
            )
            random_actions = torch.as_tensor(
                random_actions_raw[start:end], dtype=torch.float32, device=device
            )
            states_norm = torch.clamp(
                (states - rms["mean"]) / torch.sqrt(rms["var"] + 1e-8), -5.0, 5.0
            )
            actor_action = actor(states_norm)
            expert_action_norm = (expert_actions - rms["act_mean"]) / torch.sqrt(
                rms["act_var"] + 1e-8
            )
            actor_action_norm = (actor_action - rms["act_mean"]) / torch.sqrt(
                rms["act_var"] + 1e-8
            )
            random_action_norm = (random_actions - rms["act_mean"]) / torch.sqrt(
                rms["act_var"] + 1e-8
            )
            actor_actions.append(actor_action.cpu().numpy())
            expert_scores.append(
                torch.sigmoid(discriminator(states_norm, expert_action_norm)).cpu().numpy()
            )
            actor_scores.append(
                torch.sigmoid(discriminator(states_norm, actor_action_norm)).cpu().numpy()
            )
            random_scores.append(
                torch.sigmoid(discriminator(states_norm, random_action_norm)).cpu().numpy()
            )
    return (
        np.concatenate(actor_actions, axis=0),
        np.concatenate(expert_scores, axis=0).reshape(-1),
        np.concatenate(actor_scores, axis=0).reshape(-1),
        np.concatenate(random_scores, axis=0).reshape(-1),
    )


def _save_sample_table(
    output_path: Path,
    source_rows: np.ndarray,
    episode_labels: np.ndarray,
    states: np.ndarray,
    expert_actions: np.ndarray,
    actor_actions: np.ndarray,
    random_actions: np.ndarray,
    expert_scores: np.ndarray,
    actor_scores: np.ndarray,
    random_scores: np.ndarray,
) -> None:
    columns: dict[str, Any] = {
        "source_row": source_rows,
        "expert_episode_id": episode_labels,
    }
    for index in range(states.shape[1]):
        columns[f"state_{index}"] = states[:, index]
    for index in range(expert_actions.shape[1]):
        columns[f"expert_action_{index}"] = expert_actions[:, index]
        columns[f"actor_action_{index}"] = actor_actions[:, index]
        columns[f"random_action_{index}"] = random_actions[:, index]
    columns["expert_score"] = expert_scores
    columns["actor_score"] = actor_scores
    columns["random_score"] = random_scores
    pd.DataFrame(columns).to_csv(output_path, index=False, encoding="utf-8-sig")


def _plot_seed(
    seed: int,
    expert_scores: np.ndarray,
    actor_scores: np.ndarray,
    random_scores: np.ndarray,
    metrics: dict[str, Any],
    output_path: Path,
) -> None:
    actor_metrics = metrics["actor_vs_expert"]
    random_metrics = metrics["random_vs_expert"]
    figure, axis = plt.subplots(figsize=(11.3, 6.4))
    axis.hist(
        expert_scores,
        bins=50,
        range=(0, 1),
        alpha=0.50,
        color="#2ca02c",
        label=f"Expert (Mean={metrics['expert']['mean']:.3f}, n={metrics['expert']['count']})",
    )
    axis.hist(
        actor_scores,
        bins=50,
        range=(0, 1),
        alpha=0.45,
        color="#d62728",
        label=(
            f"Actor (Mean={actor_metrics['target_mean']:.3f}, "
            f"JS={actor_metrics['js_divergence']:.3f}, "
            f"AUC={actor_metrics['roc_auc_expert_positive']:.3f}, "
            f"P_eq={actor_metrics['mean_equivalence_p_value']:.2e})"
        ),
    )
    axis.hist(
        random_scores,
        bins=50,
        range=(0, 1),
        alpha=0.28,
        color="#7f7f7f",
        label=(
            f"Random (Mean={random_metrics['target_mean']:.3f}, "
            f"JS={random_metrics['js_divergence']:.3f}, "
            f"AUC={random_metrics['roc_auc_expert_positive']:.3f}, "
            f"P_diff={random_metrics['paired_permutation_p_value']:.2e})"
        ),
    )
    axis.axvline(metrics["expert"]["mean"], color="#1b6e1b", linestyle="--", linewidth=1.5)
    axis.axvline(actor_metrics["target_mean"], color="#8b1a1a", linestyle="--", linewidth=1.5)
    axis.axvline(random_metrics["target_mean"], color="#555555", linestyle="--", linewidth=1.5)
    axis.set_title(f"Held-out Online-GAIL Discriminator Evaluation (seed={seed})")
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
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    expert_csv: Path = DEFAULT_EXPERT_CSV,
    *,
    bins: int = 100,
    bootstrap_iterations: int = 500,
    permutation_iterations: int = 5000,
    equivalence_margin: float = 0.05,
    auc_equivalence_margin: float = 0.05,
    inference_batch_size: int = 4096,
    action_bound: float = 0.5,
    metric_random_seed: int = 20260728,
    device_name: str | None = None,
) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_root) / f"seed_{seed}"
    output_dir = Path(output_root) / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    actor_path = checkpoint_dir / "production1_actor.pth"
    discriminator_path = checkpoint_dir / "discriminator.pth"
    rms_path = checkpoint_dir / "obs_rms_params.pth"
    split_manifest_path = checkpoint_dir / "expert_split.json"
    required = (
        actor_path,
        discriminator_path,
        rms_path,
        split_manifest_path,
        Path(expert_csv),
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing evaluation inputs for seed {seed}: {missing}")

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    actor = FinalActor(s_dim=33, a_dim=4, a_bound=action_bound).to(device)
    discriminator = FinalDiscriminator(s_dim=33, a_dim=4).to(device)
    actor.load_state_dict(_load_state_dict(actor_path, device))
    discriminator.load_state_dict(_load_state_dict(discriminator_path, device))
    actor.eval()
    discriminator.eval()
    rms = _load_rms(rms_path, device)

    source_rows, episode_labels, evaluation_data, split_metadata = (
        _load_held_out_expert_data(Path(expert_csv), split_manifest_path)
    )
    states_raw = evaluation_data[:, :33]
    expert_actions_raw = evaluation_data[:, 33:37]
    rng = np.random.default_rng(metric_random_seed + int(seed))
    random_actions_raw = rng.uniform(
        -action_bound, action_bound, size=expert_actions_raw.shape
    ).astype(np.float32)
    actor_actions_raw, expert_scores, actor_scores, random_scores = _score_actions(
        actor,
        discriminator,
        states_raw,
        expert_actions_raw,
        random_actions_raw,
        rms,
        device,
        inference_batch_size,
    )

    actor_metrics = compare_distributions(
        expert_scores,
        actor_scores,
        episode_labels,
        bins=bins,
        bootstrap_iterations=bootstrap_iterations,
        auc_equivalence_margin=auc_equivalence_margin,
        rng=rng,
    )
    random_metrics = compare_distributions(
        expert_scores,
        random_scores,
        episode_labels,
        bins=bins,
        bootstrap_iterations=bootstrap_iterations,
        auc_equivalence_margin=auc_equivalence_margin,
        rng=rng,
    )
    expert_episode_means = _episode_means(expert_scores, episode_labels)
    actor_episode_means = _episode_means(actor_scores, episode_labels)
    random_episode_means = _episode_means(random_scores, episode_labels)
    actor_metrics.update(
        _paired_tost(expert_episode_means, actor_episode_means, equivalence_margin)
    )
    actor_metrics["indistinguishable_by_prespecified_score_criteria"] = bool(
        actor_metrics["mean_equivalent_at_alpha_0_05"]
        and actor_metrics["roc_auc_equivalent_ci90"]
    )
    random_metrics.update(
        _paired_permutation_greater(
            expert_episode_means,
            random_episode_means,
            permutation_iterations,
            rng,
        )
    )
    random_metrics["distinguishable_by_prespecified_score_criteria"] = bool(
        random_metrics["expert_score_greater_at_alpha_0_05"]
        and random_metrics["roc_auc_ci95_low"] > 0.5
    )

    metrics: dict[str, Any] = {
        "seed": int(seed),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "expert_csv": str(Path(expert_csv).resolve()),
        "evaluation_partition": "held_out_complete_expert_episodes",
        "action_bound": float(action_bound),
        "histogram_bins_for_js": int(bins),
        "metric_random_seed": int(metric_random_seed + int(seed)),
        "statistical_unit_for_inference": "held_out_expert_episode",
        "score_pairing": "same_held_out_expert_state_across_all_action_sources",
        "actor_primary_test": (
            "paired TOST on episode-mean score difference with equivalence margin "
            f"+/-{equivalence_margin:.6g}"
        ),
        "random_primary_test": (
            "one-sided paired sign-flip permutation test on episode-mean score difference"
        ),
        "expert_split": split_metadata,
        "expert": {
            "count": int(expert_scores.size),
            "episode_count": int(np.unique(episode_labels).size),
            "mean": float(np.mean(expert_scores)),
            "std": float(np.std(expert_scores, ddof=1)) if expert_scores.size > 1 else 0.0,
        },
        "actor_vs_expert": actor_metrics,
        "random_vs_expert": random_metrics,
        "checkpoint_sha256": {
            "production1_actor.pth": _sha256(actor_path),
            "discriminator.pth": _sha256(discriminator_path),
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
        actor_actions_raw,
        random_actions_raw,
        expert_scores,
        actor_scores,
        random_scores,
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, ensure_ascii=False, indent=2)
    flat_metrics = {
        "seed": seed,
        "expert_count": metrics["expert"]["count"],
        "expert_episode_count": metrics["expert"]["episode_count"],
        "expert_mean": metrics["expert"]["mean"],
        "actor_mean": actor_metrics["target_mean"],
        "actor_js": actor_metrics["js_divergence"],
        "actor_js_ci95_low": actor_metrics["js_divergence_ci95_low"],
        "actor_js_ci95_high": actor_metrics["js_divergence_ci95_high"],
        "actor_auc": actor_metrics["roc_auc_expert_positive"],
        "actor_auc_ci90_low": actor_metrics["roc_auc_ci90_low"],
        "actor_auc_ci90_high": actor_metrics["roc_auc_ci90_high"],
        "actor_auc_equivalent_ci90": actor_metrics["roc_auc_equivalent_ci90"],
        "actor_mean_difference": actor_metrics["paired_mean_score_difference"],
        "actor_p_equivalence": actor_metrics["mean_equivalence_p_value"],
        "actor_mean_equivalent": actor_metrics["mean_equivalent_at_alpha_0_05"],
        "actor_indistinguishable_score_criteria": actor_metrics[
            "indistinguishable_by_prespecified_score_criteria"
        ],
        "actor_wasserstein": actor_metrics["wasserstein_distance"],
        "random_mean": random_metrics["target_mean"],
        "random_js": random_metrics["js_divergence"],
        "random_js_ci95_low": random_metrics["js_divergence_ci95_low"],
        "random_js_ci95_high": random_metrics["js_divergence_ci95_high"],
        "random_auc": random_metrics["roc_auc_expert_positive"],
        "random_auc_ci95_low": random_metrics["roc_auc_ci95_low"],
        "random_auc_ci95_high": random_metrics["roc_auc_ci95_high"],
        "random_mean_difference": random_metrics["paired_mean_score_difference"],
        "random_p_difference": random_metrics["paired_permutation_p_value"],
        "random_expert_score_greater": random_metrics[
            "expert_score_greater_at_alpha_0_05"
        ],
        "random_distinguishable_score_criteria": random_metrics[
            "distinguishable_by_prespecified_score_criteria"
        ],
        "random_wasserstein": random_metrics["wasserstein_distance"],
    }
    pd.DataFrame([flat_metrics]).to_csv(
        output_dir / "metrics.csv", index=False, encoding="utf-8-sig"
    )
    _plot_seed(
        seed,
        expert_scores,
        actor_scores,
        random_scores,
        metrics,
        output_dir / "discriminator_score_distributions.png",
    )
    return metrics


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    sorted_values = values[order]
    count = len(values)
    adjusted_sorted = np.maximum.accumulate(
        [(count - index) * value for index, value in enumerate(sorted_values)]
    )
    adjusted = np.empty(count, dtype=np.float64)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def write_summary(output_root: Path, seeds: tuple[int, ...] | list[int]) -> Path:
    rows: list[dict[str, Any]] = []
    score_tables: dict[int, pd.DataFrame] = {}
    for seed in seeds:
        seed_dir = Path(output_root) / f"seed_{seed}"
        metrics_path = seed_dir / "metrics.csv"
        scores_path = seed_dir / "evaluation_samples_and_scores.csv"
        if not metrics_path.exists() or not scores_path.exists():
            continue
        rows.append(pd.read_csv(metrics_path).iloc[0].to_dict())
        score_tables[int(seed)] = pd.read_csv(scores_path)
    if not rows:
        raise FileNotFoundError(f"No per-seed evaluation results under {output_root}.")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    frame["actor_p_equivalence_holm"] = _holm_adjust(
        frame["actor_p_equivalence"].to_numpy(dtype=np.float64)
    )
    frame["random_p_difference_holm"] = _holm_adjust(
        frame["random_p_difference"].to_numpy(dtype=np.float64)
    )
    frame["actor_indistinguishable_score_criteria_holm"] = (
        (frame["actor_p_equivalence_holm"] < 0.05)
        & frame["actor_auc_equivalent_ci90"].astype(bool)
    )
    frame["random_distinguishable_score_criteria_holm"] = (
        (frame["random_p_difference_holm"] < 0.05)
        & (frame["random_auc_ci95_low"] > 0.5)
    )
    summary_path = output_root / "three_seed_metrics_summary.csv"
    frame.to_csv(summary_path, index=False, encoding="utf-8-sig")

    ordered_seeds = [seed for seed in seeds if int(seed) in score_tables]
    figure, axes = plt.subplots(
        1,
        len(ordered_seeds),
        figsize=(6.5 * len(ordered_seeds), 5.5),
        sharex=True,
    )
    if len(ordered_seeds) == 1:
        axes = [axes]
    for axis, seed in zip(axes, ordered_seeds):
        table = score_tables[int(seed)]
        row = frame.loc[frame["seed"].astype(int) == int(seed)].iloc[0]
        axis.hist(
            table["expert_score"],
            bins=50,
            range=(0, 1),
            alpha=0.50,
            color="#2ca02c",
            label=f"Expert (Mean={row['expert_mean']:.3f})",
        )
        axis.hist(
            table["actor_score"],
            bins=50,
            range=(0, 1),
            alpha=0.45,
            color="#d62728",
            label=(
                f"Actor (Mean={row['actor_mean']:.3f}, JS={row['actor_js']:.3f}, "
                f"AUC={row['actor_auc']:.3f}, P_eq={row['actor_p_equivalence']:.2e})"
            ),
        )
        axis.hist(
            table["random_score"],
            bins=50,
            range=(0, 1),
            alpha=0.28,
            color="#7f7f7f",
            label=(
                f"Random (Mean={row['random_mean']:.3f}, JS={row['random_js']:.3f}, "
                f"AUC={row['random_auc']:.3f}, P_diff={row['random_p_difference']:.2e})"
            ),
        )
        axis.set_title(f"seed={int(seed)}")
        axis.set_xlabel("Discriminator confidence score")
        axis.set_xlim(0.0, 1.0)
        axis.grid(axis="y", linestyle="--", alpha=0.25)
        axis.legend(fontsize=7.5, framealpha=0.95)
    axes[0].set_ylabel("Sample count")
    figure.suptitle("Held-out Online-GAIL Discriminator Evaluation", y=1.02)
    figure.tight_layout()
    figure.savefig(
        output_root / "three_seed_discriminator_score_distributions.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--expert-csv", type=Path, default=DEFAULT_EXPERT_CSV)
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    parser.add_argument("--permutation-iterations", type=int, default=5000)
    parser.add_argument("--equivalence-margin", type=float, default=0.05)
    parser.add_argument("--auc-equivalence-margin", type=float, default=0.05)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(args.seeds)
    for seed in seeds:
        metrics = evaluate_seed(
            seed,
            checkpoint_root=args.checkpoint_root,
            output_root=args.output_root,
            expert_csv=args.expert_csv,
            bootstrap_iterations=args.bootstrap_iterations,
            permutation_iterations=args.permutation_iterations,
            equivalence_margin=args.equivalence_margin,
            auc_equivalence_margin=args.auc_equivalence_margin,
            device_name=args.device,
        )
        print(
            f"seed={seed}: expert mean={metrics['expert']['mean']:.4f}, "
            f"actor mean={metrics['actor_vs_expert']['target_mean']:.4f}, "
            f"actor JS={metrics['actor_vs_expert']['js_divergence']:.4f}, "
            f"actor P_eq={metrics['actor_vs_expert']['mean_equivalence_p_value']:.4e}, "
            f"random mean={metrics['random_vs_expert']['target_mean']:.4f}, "
            f"random JS={metrics['random_vs_expert']['js_divergence']:.4f}, "
            f"random P_diff={metrics['random_vs_expert']['paired_permutation_p_value']:.4e}"
        )
    summary_path = write_summary(args.output_root, seeds)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
