from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from real_System_remake.Bank_config import Bank_config
from real_System_remake.Enterprise_config import Enterprise_config
import real_System_remake.Environment as environment_module
from real_System_remake.Environment import Environment


BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_ROOT = BASE_DIR / "checkpoints" / "final_weights"
OUTPUT_DIR = BASE_DIR / "analysis_plots" / "post_training_dscr_100_episodes"
EPISODES = 100

METHODS = [
    {
        "name": "TD3",
        "label": "TD3",
        "seed": 739,
        "checkpoint": CHECKPOINT_ROOT / "TD3" / "seed_739",
        "color": "#82aee8",
        "production_normalized": False,
        "production_dscr_divisor": 2.0,
    },
    {
        "name": "GAIL+TD3",
        "label": "GAIL+TD3",
        "seed": 739,
        "checkpoint": CHECKPOINT_ROOT / "GAIL+TD3" / "seed_739",
        "color": "#ef999d",
        "production_normalized": True,
        "production_dscr_divisor": 1.0,
    },
    {
        "name": "Transformer+GAIL+TD3",
        "label": "Transformer+GAIL+TD3",
        "seed": 291,
        "checkpoint": CHECKPOINT_ROOT / "Transformer+GAIL+TD3" / "seed_291",
        "color": "#91d0ac",
        "production_normalized": True,
        "production_dscr_divisor": 1.0,
    },
]


class EvaluationActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_bound: float,
        max_seq_len: int = 1,
        hist_hidden: int = 64,
        nhead: int = 4,
        use_history: bool = False,
        include_history_modules: bool = False,
    ):
        super().__init__()
        self.use_history = use_history
        self.include_history_modules = include_history_modules
        self.action_bound = action_bound

        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, 32)
        self.l3 = nn.Linear(32, action_dim)

        if include_history_modules:
            self.hist_input = nn.Linear(state_dim, hist_hidden)
            self.hist_pos_embed = nn.Parameter(
                torch.zeros(1, max_seq_len, hist_hidden)
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hist_hidden,
                nhead=nhead,
                dim_feedforward=hist_hidden * 2,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
            )
            self.hist_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=1,
            )
            self.hist_norm = nn.LayerNorm(hist_hidden)
            self.hl1 = nn.Linear(state_dim + hist_hidden, 128)
            self.hl2 = nn.Linear(128, 32)
            self.hl3 = nn.Linear(32, action_dim)

    def forward(
        self,
        state: torch.Tensor,
        history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        state = torch.as_tensor(state, dtype=torch.float32)
        if self.use_history and history is not None:
            history = torch.as_tensor(history, dtype=torch.float32)
            sequence_length = history.size(1)
            encoded = (
                self.hist_input(history)
                + self.hist_pos_embed[:, :sequence_length, :]
            )
            encoded = self.hist_encoder(encoded)
            feature = self.hist_norm(encoded[:, -1, :])
            combined = torch.cat([state, feature], dim=1)
            action = torch.tanh(self.hl1(combined))
            action = F.leaky_relu(self.hl2(action))
            return self.action_bound * torch.tanh(self.hl3(action))

        action = torch.tanh(self.l1(state))
        action = F.leaky_relu(self.l2(action))
        return self.action_bound * torch.tanh(self.l3(action))


class DeterministicPolicy:
    def __init__(
        self,
        actor: EvaluationActor,
        normalize: bool = False,
        normalization: Optional[dict] = None,
        history_length: int = 1,
        use_history: bool = False,
        bank_action: bool = False,
    ):
        self.actor = actor.eval()
        self.normalize = normalize
        self.normalization = normalization
        self.history_length = history_length
        self.use_history = use_history
        self.bank_action = bank_action
        self.history: List[np.ndarray] = []

    def reset(self) -> None:
        self.history = []

    def _actor_state(self, state) -> np.ndarray:
        state_array = np.asarray(state, dtype=np.float32)
        if self.normalize:
            mean = self.normalization["mean"]
            variance = self.normalization["var"]
            state_array = (state_array - mean) / np.sqrt(variance + 1e-8)
            state_array = np.clip(state_array, -5.0, 5.0)
        return state_array

    def action(self, state) -> np.ndarray:
        actor_state = self._actor_state(state)
        history_tensor = None
        if self.use_history:
            window = self.history + [actor_state]
            if len(window) < self.history_length:
                window = [window[0]] * (self.history_length - len(window)) + window
            else:
                window = window[-self.history_length :]
            history_tensor = torch.as_tensor(
                np.stack(window, axis=0)[np.newaxis, :, :],
                dtype=torch.float32,
            )

        with torch.no_grad():
            action = self.actor(
                torch.as_tensor(actor_state[np.newaxis, :], dtype=torch.float32),
                history_tensor,
            )[0].cpu().numpy()

        if self.use_history:
            self.history.append(actor_state.copy())
            if len(self.history) > self.history_length:
                self.history = self.history[-self.history_length :]

        if self.bank_action:
            action = action + 0.5
        return action


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def state_dict_shape(state_dict: dict, key: str) -> tuple:
    return tuple(state_dict[key].shape)


def load_actor(
    path: Path,
    use_history: bool,
) -> tuple[EvaluationActor, int]:
    state_dict = torch.load(path, map_location="cpu")
    state_dim = state_dict_shape(state_dict, "l1.weight")[1]
    action_dim = state_dict_shape(state_dict, "l3.weight")[0]
    include_history = "hist_pos_embed" in state_dict
    history_length = (
        state_dict_shape(state_dict, "hist_pos_embed")[1]
        if include_history
        else 1
    )
    hist_hidden = (
        state_dict_shape(state_dict, "hist_pos_embed")[2]
        if include_history
        else 64
    )
    nhead = 1
    if include_history:
        in_projection = state_dict_shape(
            state_dict,
            "hist_encoder.layers.0.self_attn.in_proj_weight",
        )[0]
        if hist_hidden == 64 and in_projection == 192:
            nhead = 4

    actor = EvaluationActor(
        state_dim=state_dim,
        action_dim=action_dim,
        action_bound=0.5,
        max_seq_len=history_length,
        hist_hidden=hist_hidden,
        nhead=nhead,
        use_history=use_history,
        include_history_modules=include_history,
    )
    actor.load_state_dict(state_dict, strict=True)
    return actor, history_length


def load_normalization(checkpoint_dir: Path) -> dict:
    values = torch.load(
        checkpoint_dir / "obs_rms_params.pth",
        map_location="cpu",
    )
    return {
        "mean": values["mean"].detach().cpu().numpy().astype(np.float32),
        "var": values["var"].detach().cpu().numpy().astype(np.float32),
    }


def config_section(config: dict, *names: str) -> dict:
    for name in names:
        value = config.get(name)
        if isinstance(value, dict):
            return value
    return {}


def create_environment(method: dict) -> Environment:
    checkpoint_dir = method["checkpoint"]
    config = load_json(checkpoint_dir / "config.json")

    enterprise_section = config_section(
        config,
        "enterprise_environment_config",
        "enterprise_config",
        "enterprise_env_config",
    )
    if "base_config" in enterprise_section:
        enterprise_section = enterprise_section["base_config"]
    bank_section = config_section(
        config,
        "bank_environment_config",
        "bank_config",
        "bank_env_config",
    )

    environment_module.swanlab.init = lambda *args, **kwargs: None
    environment_module.swanlab.finish = lambda *args, **kwargs: None

    environment = Environment(
        name=f"post_training_eval_{method['name']}_seed_{method['seed']}",
        lim_day=100,
    )
    for enterprise_name, output_name in (
        ("production1", "K"),
        ("consumption1", "L"),
    ):
        enterprise_config = Enterprise_config(
            name=enterprise_name,
            output_name=output_name,
            money=float(enterprise_section.get("money", 0.0)),
            WNDF=float(enterprise_section.get("WNDF", 100.0)),
            stock=float(enterprise_section.get("stock", 10.0)),
            price=float(enterprise_section.get("price", 8.0)),
            intention=float(enterprise_section.get("intention", 5.0)),
            gamma=float(enterprise_section.get("gamma", 0.95)),
        )
        environment.add_enterprise_agent(config=enterprise_config)

    bank_config = Bank_config(
        name="bank1",
        fund=float(bank_section.get("fund", 2000)),
        fund_rate=float(bank_section.get("fund_rate", 1)),
        fund_increase=float(bank_section.get("fund_increase", 0.1)),
        debt_time=int(bank_section.get("debt_time", 5)),
        debt_i=float(bank_section.get("debt_i", 0.005)),
    )
    environment.add_bank(bank_config)
    environment.add_enterprise_thirdmarket(
        name="production_thirdMarket",
        output_name="K",
        price=100,
    )
    environment.add_enterprise_thirdmarket(
        name="consumption_thirdMarket",
        output_name="L",
        price=100,
    )
    environment.init()
    environment.use_swanlab = False

    no_op = lambda *args, **kwargs: None
    for method_name in (
        "receive_action",
        "receive_enterprise",
        "receive_bank",
        "receive_daily_trajectory",
        "receive_finish_enterprise",
        "receive_finish_bank",
        "swanlab_log",
        "output_to_txt",
        "output_config",
    ):
        setattr(environment.logger, method_name, no_op)
    return environment


def create_policies(method: dict) -> Dict[str, DeterministicPolicy]:
    checkpoint_dir = method["checkpoint"]
    is_transformer = method["name"] == "Transformer+GAIL+TD3"
    normalization = (
        load_normalization(checkpoint_dir)
        if method["production_normalized"]
        else None
    )

    production_actor, production_history = load_actor(
        checkpoint_dir / "production1_actor.pth",
        use_history=is_transformer,
    )
    consumption_actor, consumption_history = load_actor(
        checkpoint_dir / "consumption1_actor.pth",
        use_history=False,
    )
    bank_actor, bank_history = load_actor(
        checkpoint_dir / "bank1_actor.pth",
        use_history=False,
    )
    return {
        "production1": DeterministicPolicy(
            production_actor,
            normalize=method["production_normalized"],
            normalization=normalization,
            history_length=production_history,
            use_history=is_transformer,
        ),
        "consumption1": DeterministicPolicy(
            consumption_actor,
            history_length=consumption_history,
        ),
        "bank1": DeterministicPolicy(
            bank_actor,
            history_length=bank_history,
            bank_action=True,
        ),
    }


def evaluate_method(method: dict) -> List[dict]:
    seed_everything(method["seed"])
    policies = create_policies(method)
    environment = create_environment(method)
    observations: List[dict] = []

    with contextlib.redirect_stdout(io.StringIO()):
        for episode in range(EPISODES):
            state = environment.reset()
            for policy in policies.values():
                policy.reset()
            previous_count = 0

            while True:
                actions = {
                    name: policy.action(state[name])
                    for name, policy in policies.items()
                }
                environment.step(actions)
                enterprise = environment.Enterprise["production1"]
                if enterprise.dscr_count > previous_count:
                    observations.append(
                        {
                            "method": method["name"],
                            "seed": method["seed"],
                            "episode": episode,
                            "day": environment.day,
                            "raw_dscr": float(enterprise.dscr),
                            "dscr": float(enterprise.dscr)
                            / method["production_dscr_divisor"],
                        }
                    )
                    previous_count = enterprise.dscr_count

                next_state, _reward, done = environment.observe()
                if done:
                    break
                state = next_state

    environment.finish()
    return observations


def percentile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability, method="linear"))


def summarize(method: dict, observations: List[dict]) -> dict:
    values = np.asarray([row["dscr"] for row in observations], dtype=float)
    q1 = percentile(values, 0.25)
    median = percentile(values, 0.5)
    q3 = percentile(values, 0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    retained = values[(values >= lower_fence) & (values <= upper_fence)]
    below_one_count = int(np.sum(values < 1.0))
    return {
        "method": method["name"],
        "label": method["label"],
        "seed": method["seed"],
        "episodes": EPISODES,
        "valid_firm_days": int(values.size),
        "mean": float(np.mean(values)),
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
        "lower_whisker": float(np.min(retained)),
        "upper_whisker": float(np.max(retained)),
        "retained_count": int(retained.size),
        "outlier_count": int(values.size - retained.size),
        "below_one_count": below_one_count,
        "below_one_share_pct": 100.0 * below_one_count / values.size,
        "color": method["color"],
    }


def write_observations(rows: List[dict]) -> Path:
    output_path = OUTPUT_DIR / "production_dscr_firm_day_observations.csv"
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "method",
                "seed",
                "episode",
                "day",
                "raw_dscr",
                "dscr",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def write_summary(rows: List[dict]) -> Path:
    output_path = OUTPUT_DIR / "production_dscr_100_episode_summary.csv"
    fieldnames = [
        key for key in rows[0].keys() if key not in ("color",)
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})
    return output_path


def draw_figure(summaries: List[dict]) -> Path:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.linewidth": 1.0,
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    box_axis, share_axis = axes

    box_stats = []
    colors = []
    labels = []
    for row in summaries:
        box_stats.append(
            {
                "label": row["label"],
                "med": row["median"],
                "q1": row["q1"],
                "q3": row["q3"],
                "whislo": row["lower_whisker"],
                "whishi": row["upper_whisker"],
                "fliers": [],
            }
        )
        colors.append(row["color"])
        labels.append(row["label"])

    artists = box_axis.bxp(
        box_stats,
        showfliers=False,
        patch_artist=True,
        widths=0.48,
        boxprops={"linewidth": 1.2},
        whiskerprops={"linewidth": 1.2, "color": "#505965"},
        capprops={"linewidth": 1.2, "color": "#505965"},
        medianprops={"linewidth": 1.8, "color": "#202a35"},
    )
    for patch, color in zip(artists["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.58)

    box_axis.axhline(
        1.0,
        color="#657487",
        linewidth=1.2,
        linestyle="--",
    )
    box_axis.text(
        0.985,
        1.02,
        "DSCR = 1",
        transform=box_axis.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#596779",
    )
    box_axis.set_ylabel("DSCR")
    box_axis.set_title("(a)", loc="left", fontweight="bold")
    box_axis.grid(axis="y", linestyle="-", linewidth=0.55, alpha=0.25)
    box_axis.spines["top"].set_visible(False)
    box_axis.spines["right"].set_visible(False)

    shares = [row["below_one_share_pct"] for row in summaries]
    bars = share_axis.bar(
        np.arange(len(summaries)),
        shares,
        color=colors,
        width=0.55,
    )
    share_axis.set_xticks(np.arange(len(summaries)), labels)
    share_axis.set_ylabel("Share (%)")
    share_axis.set_title("(b)", loc="left", fontweight="bold")
    share_axis.grid(axis="y", linestyle="-", linewidth=0.55, alpha=0.25)
    share_axis.spines["top"].set_visible(False)
    share_axis.spines["right"].set_visible(False)
    share_axis.set_ylim(0, max(max(shares) * 1.22, 1.0))
    for bar, share in zip(bars, shares):
        share_axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + share_axis.get_ylim()[1] * 0.02,
            f"{share:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.subplots_adjust(
        left=0.08,
        right=0.985,
        bottom=0.18,
        top=0.91,
        wspace=0.22,
    )
    output_path = OUTPUT_DIR / "post_training_production_dscr_comparison.png"
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)
    return output_path


def write_method_note() -> Path:
    output_path = OUTPUT_DIR / "method.txt"
    text = (
        "Post-training production-enterprise DSCR evaluation\n\n"
        "Each saved policy is evaluated deterministically for 100 episodes. "
        "Exploration noise, replay-buffer writes, and learning updates are disabled.\n"
        "TD3 uses DSCR = money / (2 * (should_payback + iDebt)).\n"
        "GAIL+TD3 and Transformer+GAIL+TD3 use the original "
        "DSCR = money / (should_payback + iDebt).\n"
        "Only firm-day observations for which the environment increments "
        "dscr_count are included.\n"
        "Panel (a) uses Q1, median, Q3, and whiskers defined as the most "
        "extreme observations inside Q1 - 1.5*IQR and Q3 + 1.5*IQR. "
        "Outliers are not drawn.\n"
        "Panel (b) reports the percentage of all valid firm-day observations "
        "with DSCR < 1 before IQR exclusion.\n"
    )
    output_path.write_text(text, encoding="utf-8")
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_observations: List[dict] = []
    summaries: List[dict] = []

    for method in METHODS:
        print(
            f"Evaluating {method['name']} seed {method['seed']} "
            f"for {EPISODES} episodes...",
            flush=True,
        )
        observations = evaluate_method(method)
        all_observations.extend(observations)
        summaries.append(summarize(method, observations))
        print(
            f"  valid firm-days={len(observations)}, "
            f"DSCR<1={summaries[-1]['below_one_share_pct']:.2f}%",
            flush=True,
        )

    observation_path = write_observations(all_observations)
    summary_path = write_summary(summaries)
    figure_path = draw_figure(summaries)
    method_path = write_method_note()

    print(observation_path.resolve())
    print(summary_path.resolve())
    print(figure_path.resolve())
    print(method_path.resolve())


if __name__ == "__main__":
    main()
