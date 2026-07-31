"""Evaluate historical deterministic Actor snapshots with a fixed discriminator.

For each seed, every Actor snapshot is evaluated on the same held-out expert
states.  The expert actions, state normalization parameters, and selected final
discriminator are fixed within that seed.  Consequently, changes in the output
metrics are attributable to changes in the saved Actor parameters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import torch

try:
    from real_System_remake.expert_data_split import (
        file_sha256,
        load_expert_episode_split,
    )
    from real_System_remake.final_gail_statistical_eval import (
        FinalActor,
        FinalDiscriminator,
        _js_divergence,
        _load_rms,
        _load_state_dict,
        _roc_auc,
    )
except ModuleNotFoundError:
    from expert_data_split import file_sha256, load_expert_episode_split
    from final_gail_statistical_eval import (
        FinalActor,
        FinalDiscriminator,
        _js_divergence,
        _load_rms,
        _load_state_dict,
        _roc_auc,
    )


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_ROOT = (
    PROJECT_DIR / "checkpoints" / "final_weights" / "GAIL+TD3_balanced_discriminator"
)
DEFAULT_EXPERT_CSV = PROJECT_DIR / "expert_data_production1_collected.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "analysis_plots" / "actor_snapshot_evolution"
DEFAULT_SEEDS = (184, 652, 187)


def _score_pairs(
    discriminator: FinalDiscriminator,
    states_normalized: torch.Tensor,
    actions_normalized: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    scores: list[np.ndarray] = []
    discriminator.eval()
    with torch.no_grad():
        for start in range(0, len(states_normalized), batch_size):
            end = min(start + batch_size, len(states_normalized))
            logits = discriminator(
                states_normalized[start:end], actions_normalized[start:end]
            )
            scores.append(torch.sigmoid(logits).cpu().numpy().reshape(-1))
    return np.concatenate(scores).astype(np.float64, copy=False)


def _average_ranks(values: np.ndarray) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: float(values[index]))
    ranks = [0.0] * len(order)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return ranks


def _spearman(values: np.ndarray) -> tuple[float, float]:
    indices = np.arange(1, len(values) + 1, dtype=np.float64)
    x_ranks = _average_ranks(indices)
    y_ranks = _average_ranks(values)
    x_mean = sum(x_ranks) / len(x_ranks)
    y_mean = sum(y_ranks) / len(y_ranks)
    numerator = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_ranks, y_ranks)
    )
    x_ss = sum((value - x_mean) ** 2 for value in x_ranks)
    y_ss = sum((value - y_mean) ** 2 for value in y_ranks)
    rho = numerator / (x_ss * y_ss) ** 0.5
    if len(values) > 2 and abs(rho) < 1.0:
        statistic = rho * ((len(values) - 2) / (1.0 - rho**2)) ** 0.5
        p_value = float(2.0 * student_t.sf(abs(statistic), len(values) - 2))
    else:
        p_value = 0.0 if abs(rho) == 1.0 else float("nan")
    return float(rho), p_value


def _load_fixed_inputs(
    checkpoint_dir: Path,
    expert_csv: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    discriminator_path = checkpoint_dir / "discriminator_best_validation.pth"
    rms_path = checkpoint_dir / "obs_rms_params.pth"
    split_manifest_path = checkpoint_dir / "expert_split.json"
    required = (discriminator_path, rms_path, split_manifest_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing fixed evaluation inputs: {missing}")

    split = load_expert_episode_split(expert_csv)
    saved_split = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    if split.metadata["expert_csv_sha256"] != saved_split["expert_csv_sha256"]:
        raise ValueError("The current expert CSV differs from the checkpoint expert data.")

    rms = _load_rms(rms_path, device)
    values = split.test_values
    states_raw = torch.as_tensor(values[:, :33], dtype=torch.float32, device=device)
    actions_raw = torch.as_tensor(values[:, 33:37], dtype=torch.float32, device=device)
    states_normalized = torch.clamp(
        (states_raw - rms["mean"]) / torch.sqrt(rms["var"] + 1e-8), -5.0, 5.0
    )
    actions_normalized = (actions_raw - rms["act_mean"]) / torch.sqrt(
        rms["act_var"] + 1e-8
    )

    discriminator = FinalDiscriminator().to(device)
    discriminator.load_state_dict(_load_state_dict(discriminator_path, device))
    expert_scores = _score_pairs(
        discriminator, states_normalized, actions_normalized, batch_size
    )
    return {
        "discriminator": discriminator,
        "discriminator_path": discriminator_path,
        "rms_path": rms_path,
        "rms": rms,
        "states_normalized": states_normalized,
        "expert_actions_raw": actions_raw,
        "expert_scores": expert_scores,
        "test_source_rows": split.test_source_rows,
        "test_episode_labels": split.test_episode_labels,
        "expert_csv_sha256": split.metadata["expert_csv_sha256"],
        "test_episode_count": int(np.unique(split.test_episode_labels).size),
    }


def evaluate_seed(
    seed: int,
    *,
    checkpoint_root: Path,
    expert_csv: Path,
    output_dir: Path,
    device: torch.device,
    bins: int,
    batch_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    checkpoint_dir = checkpoint_root / f"seed_{seed}"
    snapshot_dir = checkpoint_dir / "actor_snapshots"
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Seed {seed} has no Actor snapshot manifest. Retraining is required: "
            f"{manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshots = manifest.get("snapshots", [])
    if not snapshots:
        raise ValueError(f"Seed {seed} Actor snapshot manifest is empty.")
    evaluation_steps = [int(item["evaluation_step"]) for item in snapshots]
    if evaluation_steps != sorted(set(evaluation_steps)):
        raise ValueError(f"Seed {seed} snapshot evaluation steps are not unique and sorted.")

    fixed = _load_fixed_inputs(checkpoint_dir, expert_csv, device, batch_size)
    expert_scores = fixed["expert_scores"]
    actor_scores_matrix = np.empty(
        (len(snapshots), len(expert_scores)), dtype=np.float32
    )
    rows: list[dict[str, Any]] = []
    actor = FinalActor().to(device)
    discriminator = fixed["discriminator"]
    action_scale = torch.sqrt(fixed["rms"]["act_var"] + 1e-8)
    action_mean = fixed["rms"]["act_mean"]

    for index, snapshot in enumerate(snapshots):
        actor_path = snapshot_dir / snapshot["actor_checkpoint"]
        if not actor_path.is_file():
            raise FileNotFoundError(f"Missing Actor snapshot: {actor_path}")
        actor.load_state_dict(_load_state_dict(actor_path, device))
        actor.eval()
        with torch.no_grad():
            actor_actions = actor(fixed["states_normalized"])
            actor_actions_normalized = (actor_actions - action_mean) / action_scale
        actor_scores = _score_pairs(
            discriminator,
            fixed["states_normalized"],
            actor_actions_normalized,
            batch_size,
        )
        actor_scores_matrix[index] = actor_scores.astype(np.float32)
        action_error = actor_actions - fixed["expert_actions_raw"]
        rows.append(
            {
                "seed": int(seed),
                "evaluation_step": int(snapshot["evaluation_step"]),
                "completed_environment_episode": int(
                    snapshot["completed_environment_episode"]
                ),
                "td3_parameter_update_step": int(
                    snapshot["td3_parameter_update_step"]
                ),
                "total_sim_days": int(snapshot["total_sim_days"]),
                "expert_sample_count": int(len(expert_scores)),
                "expert_score_mean": float(expert_scores.mean()),
                "actor_score_mean": float(actor_scores.mean()),
                "expert_minus_actor_score_mean": float(
                    expert_scores.mean() - actor_scores.mean()
                ),
                "actor_expert_js_divergence": _js_divergence(
                    expert_scores, actor_scores, bins
                ),
                "expert_vs_actor_roc_auc": _roc_auc(
                    expert_scores, actor_scores
                ),
                "auc_distance_from_0_5": abs(
                    _roc_auc(expert_scores, actor_scores) - 0.5
                ),
                "action_mae": float(action_error.abs().mean().cpu()),
                "action_rmse": float(
                    torch.sqrt(torch.mean(action_error.pow(2))).cpu()
                ),
                "actor_checkpoint": str(actor_path.resolve()),
                "actor_checkpoint_sha256": file_sha256(actor_path),
            }
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / f"seed_{seed}_actor_snapshot_metrics.csv", index=False)
    np.savez_compressed(
        output_dir / f"seed_{seed}_actor_snapshot_scores.npz",
        evaluation_steps=metrics["evaluation_step"].to_numpy(dtype=np.int32),
        expert_scores=expert_scores.astype(np.float32),
        actor_scores=actor_scores_matrix,
        expert_test_source_rows=fixed["test_source_rows"].astype(np.int64),
        expert_test_episode_labels=fixed["test_episode_labels"].astype(np.int64),
    )

    js_values = metrics["actor_expert_js_divergence"].to_numpy(dtype=np.float64)
    auc_distances = metrics["auc_distance_from_0_5"].to_numpy(dtype=np.float64)
    edge_count = max(1, int(np.ceil(len(metrics) * 0.20)))
    js_rho, js_p = _spearman(js_values)
    auc_rho, auc_p = _spearman(auc_distances)
    summary = {
        "seed": int(seed),
        "snapshot_count": int(len(metrics)),
        "first_evaluation_step": int(metrics["evaluation_step"].iloc[0]),
        "last_evaluation_step": int(metrics["evaluation_step"].iloc[-1]),
        "expert_test_rows": int(len(expert_scores)),
        "expert_test_episodes": fixed["test_episode_count"],
        "early_snapshot_count": int(edge_count),
        "early_js_mean": float(js_values[:edge_count].mean()),
        "late_js_mean": float(js_values[-edge_count:].mean()),
        "late_minus_early_js": float(
            js_values[-edge_count:].mean() - js_values[:edge_count].mean()
        ),
        "js_spearman_rho": js_rho,
        "js_spearman_two_sided_p_value": js_p,
        "js_significant_downward_trend": bool(js_rho < 0.0 and js_p < 0.05),
        "early_auc_distance_mean": float(auc_distances[:edge_count].mean()),
        "late_auc_distance_mean": float(auc_distances[-edge_count:].mean()),
        "auc_distance_spearman_rho": auc_rho,
        "auc_distance_spearman_two_sided_p_value": auc_p,
        "auc_distance_significant_downward_trend": bool(
            auc_rho < 0.0 and auc_p < 0.05
        ),
        "fixed_discriminator": str(fixed["discriminator_path"].resolve()),
        "fixed_discriminator_sha256": file_sha256(fixed["discriminator_path"]),
        "normalization_parameters_sha256": file_sha256(fixed["rms_path"]),
        "expert_csv_sha256": fixed["expert_csv_sha256"],
    }
    return metrics, summary


def _plot_python() -> Path:
    executable = Path(sys.executable)
    if "envs" in [part.lower() for part in executable.parts] and len(executable.parents) >= 3:
        candidate = executable.parents[2] / "python.exe"
        if candidate.exists():
            return candidate
    return executable


def evaluate_all(
    seeds: list[int] | tuple[int, ...] = DEFAULT_SEEDS,
    *,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
    expert_csv: Path = DEFAULT_EXPERT_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    device_name: str | None = None,
    bins: int = 100,
    batch_size: int = 4096,
) -> Path:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    summaries: list[dict[str, Any]] = []
    all_metrics: list[pd.DataFrame] = []
    for seed in seeds:
        print(f"[Actor evolution] evaluating seed {seed} on {device}")
        metrics, summary = evaluate_seed(
            int(seed),
            checkpoint_root=Path(checkpoint_root).resolve(),
            expert_csv=Path(expert_csv).resolve(),
            output_dir=output_dir,
            device=device,
            bins=bins,
            batch_size=batch_size,
        )
        all_metrics.append(metrics)
        summaries.append(summary)
    pd.concat(all_metrics, ignore_index=True).to_csv(
        output_dir / "three_seed_actor_snapshot_metrics.csv", index=False
    )
    pd.DataFrame(summaries).to_csv(
        output_dir / "three_seed_actor_evolution_summary.csv", index=False
    )
    methodology = {
        "evaluation_design": "historical_deterministic_actor_snapshots_under_fixed_inputs",
        "seeds": [int(seed) for seed in seeds],
        "fixed_within_seed": [
            "held-out final-test expert states",
            "recorded expert actions",
            "state and action normalization parameters",
            "validation-selected final discriminator",
        ],
        "only_changing_component": "production-enterprise Actor snapshot",
        "actor_action": "deterministic Actor output without exploration noise",
        "js_definition": (
            "Jensen-Shannon divergence over fixed-bin discriminator sigmoid-score "
            f"histograms; bins={bins}."
        ),
        "interpretation": (
            "A downward JS trajectory indicates that successive deterministic Actor "
            "outputs on the same held-out expert states become closer to recorded "
            "expert actions under the fixed discriminator's score representation."
        ),
    }
    (output_dir / "methodology.json").write_text(
        json.dumps(methodology, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plotter = PROJECT_DIR / "plot_actor_snapshot_evolution.py"
    subprocess.run(
        [
            str(_plot_python()),
            str(plotter),
            "--output-dir",
            str(output_dir),
            "--seeds",
            *[str(seed) for seed in seeds],
        ],
        check=True,
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--expert-csv", type=Path, default=DEFAULT_EXPERT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default=None)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    output_dir = evaluate_all(
        args.seeds,
        checkpoint_root=args.checkpoint_root,
        expert_csv=args.expert_csv,
        output_dir=args.output_dir,
        device_name=args.device,
        bins=args.bins,
        batch_size=args.batch_size,
    )
    print(f"Actor snapshot evolution outputs: {output_dir}")


if __name__ == "__main__":
    main()
