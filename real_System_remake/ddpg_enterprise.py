'''
ep = 企业, mod = 企业家, 一个episode = 一个企业从生到死
enterprise = enterprise_nnu(num:int)//传入企业家的个数,生成num个mod
enterprise._run_enterpeise(enterprise_mod:int, state:list)//传入经营的企业家编号&企业状态,
'''

import warnings

from Agent import Config
from Cortex import *
# from Cortex.ActorDQN import *
# from Cortex.Common.Network import leaky_relu
# from Cortex.Common.Network import AdamOptimizer
# from Cortex.Common.Network import TF_Neural_Network as Network
# from Cortex.Common.ExperienceReplay import pick_selector_class as PickSelectorClass
import warnings
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from scipy.stats import rankdata
# import tensorflow as tf
from new_calculate import *
# from Agent.DDPG import DDPG
from Agent.TD3 import TD3
try:
    from real_System_remake.expert_data_split import load_expert_episode_split
except ModuleNotFoundError:
    from expert_data_split import load_expert_episode_split

# from Agent.TD3_attention import TD3 as TD3_attn  # 如果要使用其他的算法，在import中改掉即可
# from Agent.TD3withoutNoise import TD3
warnings.filterwarnings('ignore')

io_path = 'io/'
ex_path = io_path + 'enterprise_nnu/'
logs_path = ex_path + 'logs'
session_path = ex_path + 'session'
model_filename = ex_path + 'model'

clustered_devices = None

# 1. 生成器 (Actor): 负责产生动作
class RealActor(nn.Module):
    def __init__(self, s_dim=33, a_dim=4, a_bound=0.5):
        super(RealActor, self).__init__()
        self.l1 = nn.Linear(s_dim, 128)
        self.l2 = nn.Linear(128, 32)
        self.l3 = nn.Linear(32, a_dim)
        self.a_bound = a_bound

    def forward(self, state):
        # 统一处理输入类型
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32)
        a = torch.tanh(self.l1(state))
        a = F.leaky_relu(self.l2(a))
        return self.a_bound * torch.tanh(self.l3(a))

# 2. 判别器 (Discriminator): 负责给动作“打分”
class RealDiscriminator(nn.Module):
    def __init__(self, s_dim=33, a_dim=4, hidden_size=100):
        super(RealDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + a_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1) # 输出 Logits
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

class enterprise_nnu:
    def __init__(self, config: Config):
        self.config = config
        self.scope = config.scope
        self.enterprise = TD3(config=config)  # 正常的 TD3 实例
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.scope == 'production1':
            print(f"=== {self.scope}: online GAIL+TD3 training ===")
            current_dir = os.path.dirname(os.path.abspath(__file__))

            # 保留这句提示，代表它是随机初始化的
            print("TD3 Actor is randomly initialized for online adversarial training.")

            # Load expert data for online GAIL.
            csv_path = os.path.join(current_dir, 'expert_data_production1_collected.csv')
            if os.path.exists(csv_path):
                expert_split = load_expert_episode_split(csv_path)
                expert_data = torch.as_tensor(
                    expert_split.train_values, dtype=torch.float32, device=self.device
                )
                validation_data = torch.as_tensor(
                    expert_split.validation_values, dtype=torch.float32, device=self.device
                )
                self.expert_split_metadata = expert_split.metadata
                # 切分状态与动作 (前33是状态，后4是动作)
                self.expert_states = expert_data[:, :33]
                self.expert_actions = expert_data[:, 33:37]
                self.expert_size = len(self.expert_states)
                self.validation_states_raw = validation_data[:, :33]
                self.validation_actions = validation_data[:, 33:37]
                print(
                    "Loaded episode-level expert split: "
                    f"train={self.expert_split_metadata['train_episodes']} episodes/"
                    f"{self.expert_split_metadata['train_rows']} rows, "
                    f"validation={self.expert_split_metadata['validation_episodes']} episodes/"
                    f"{self.expert_split_metadata['validation_rows']} rows, "
                    f"test={self.expert_split_metadata['test_episodes']} episodes/"
                    f"{self.expert_split_metadata['test_rows']} rows."
                )
            else:
                raise FileNotFoundError(f"❌ 找不到专家数据文件: {csv_path}")

            # Compute normalization stats from the training split only.
            self.obs_mean = self.expert_states.mean(dim=0)
            self.obs_var = torch.clamp(self.expert_states.var(dim=0, unbiased=False), min=1e-6)
            self.act_mean = self.expert_actions.mean(dim=0)
            action_variance_floor = float(config.GAIL_ACTION_STD_FLOOR) ** 2
            self.act_var = torch.clamp(
                self.expert_actions.var(dim=0, unbiased=False),
                min=action_variance_floor,
            )
            print("Online GAIL normalization stats are computed from the expert training split.")

            self.validation_states = torch.clamp(
                (self.validation_states_raw - self.obs_mean)
                / torch.sqrt(self.obs_var + 1e-8),
                -5.0,
                5.0,
            )
            validation_rng = np.random.default_rng(config.random_seed + 20260729)
            self.validation_random_actions = torch.as_tensor(
                validation_rng.uniform(
                    -float(config.action_bound),
                    float(config.action_bound),
                    size=(len(self.validation_states), config.action_dim),
                ),
                dtype=torch.float32,
                device=self.device,
            )

            recent_capacity = max(1, int(config.GAIL_RECENT_BUFFER_CAPACITY))
            self.recent_policy_states = np.empty(
                (recent_capacity, config.state_dim), dtype=np.float32
            )
            self.recent_policy_actions = np.empty(
                (recent_capacity, config.action_dim), dtype=np.float32
            )
            self.recent_policy_capacity = recent_capacity
            self.recent_policy_size = 0
            self.recent_policy_position = 0
            self.recent_policy_rng = np.random.default_rng(config.random_seed + 17)

            self.validation_start_steps = int(config.GAIL_VALIDATION_START_STEPS)
            self.validation_interval = max(1, int(config.GAIL_VALIDATION_INTERVAL))
            self.validation_random_auc_min = float(config.GAIL_VALIDATION_RANDOM_AUC_MIN)
            self.next_validation_step = self.validation_start_steps
            self.best_gail_validation_score = float('inf')
            self.best_gail_validation_rank = (2, float('inf'))
            self.best_gail_checkpoint = None
            self.gail_validation_history = []

            self.gail_disc = RealDiscriminator(s_dim=33, a_dim=4).to(self.device)
            print("GAIL discriminator is randomly initialized and trained online.")

            # 【核心改变】：解冻判别器，开启训练模式
            self.gail_disc.train()
            for param in self.gail_disc.parameters():
                param.requires_grad = True

            # 固定权重
            self.disc_optimizer = torch.optim.Adam(self.gail_disc.parameters(), lr=3e-4)

            # 加入镇静剂：降低学习率 + 添加 L2 正则化）
            # self.disc_optimizer = torch.optim.Adam(
            #     self.gail_disc.parameters(),
            #     lr=3e-5,  # 降低学习率，让它学得慢一点
            #     weight_decay=1e-2  # 核心！加入强大的权重衰减，强制打分变平滑
            # )
            # print(f"🔥 判别器已装配降智版优化器 (低LR+高Weight Decay)！")
            # # 统一命名为 last_internal_reward
            # self.last_internal_reward = 0.0

            # 将这些模块的“遥控器”交给 TD3 实例，供底层 learn() 函数使用
            self.enterprise.gail_disc = self.gail_disc
            self.enterprise.disc_optimizer = self.disc_optimizer
            self.enterprise.sample_expert = self.sample_expert
            self.enterprise.sample_recent_policy = self.sample_recent_policy
            # 把标准化参数也传进去，底层算 Loss 时需要用到
            self.enterprise.obs_mean = self.obs_mean
            self.enterprise.obs_var = self.obs_var
            self.enterprise.act_mean = self.act_mean
            self.enterprise.act_var = self.act_var

    def sample_expert(self, batch_size):
        """从专家库中随机抽取一批真数据"""
        if getattr(self, 'expert_size', 0) == 0:
            return None, None
        # 随机生成 batch_size 个索引
        indices = torch.randint(0, self.expert_size, (batch_size,), device=self.device)
        return self.expert_states[indices], self.expert_actions[indices]

    def _store_recent_policy(self, normalized_state, action):
        position = self.recent_policy_position
        self.recent_policy_states[position] = np.asarray(normalized_state, dtype=np.float32)
        self.recent_policy_actions[position] = np.asarray(action, dtype=np.float32)
        self.recent_policy_position = (position + 1) % self.recent_policy_capacity
        self.recent_policy_size = min(
            self.recent_policy_size + 1, self.recent_policy_capacity
        )

    def sample_recent_policy(self, batch_size):
        if getattr(self, 'recent_policy_size', 0) == 0:
            return None, None
        indices = self.recent_policy_rng.choice(
            self.recent_policy_size,
            size=int(batch_size),
            replace=self.recent_policy_size < int(batch_size),
        )
        states = torch.as_tensor(
            self.recent_policy_states[indices], dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            self.recent_policy_actions[indices], dtype=torch.float32, device=self.device
        )
        return states, actions

    @staticmethod
    def _roc_auc(expert_positive, target_negative):
        expert_positive = np.asarray(expert_positive, dtype=np.float64).reshape(-1)
        target_negative = np.asarray(target_negative, dtype=np.float64).reshape(-1)
        values = np.concatenate([expert_positive, target_negative])
        ranks = rankdata(values, method='average')
        n_positive = len(expert_positive)
        n_negative = len(target_negative)
        positive_rank_sum = float(ranks[:n_positive].sum())
        return (
            positive_rank_sum - n_positive * (n_positive + 1) / 2.0
        ) / (n_positive * n_negative)

    @staticmethod
    def _js_distance(first, second, bins=80):
        first_hist, edges = np.histogram(first, bins=bins, range=(0.0, 1.0))
        second_hist, _ = np.histogram(second, bins=edges)
        first_prob = first_hist.astype(np.float64) + 1e-12
        second_prob = second_hist.astype(np.float64) + 1e-12
        first_prob /= first_prob.sum()
        second_prob /= second_prob.sum()
        midpoint = 0.5 * (first_prob + second_prob)
        divergence = 0.5 * np.sum(first_prob * np.log(first_prob / midpoint))
        divergence += 0.5 * np.sum(second_prob * np.log(second_prob / midpoint))
        return float(np.sqrt(max(divergence, 0.0)))

    def _discriminator_scores(self, states, actions, batch_size=4096):
        scores = []
        with torch.no_grad():
            for start in range(0, len(states), batch_size):
                logits = self.gail_disc(
                    states[start:start + batch_size],
                    actions[start:start + batch_size],
                )
                scores.append(torch.sigmoid(logits).reshape(-1).cpu())
        return torch.cat(scores).numpy()

    @staticmethod
    def _cpu_state_dict(module):
        return {
            key: value.detach().cpu().clone()
            for key, value in module.state_dict().items()
        }

    def _evaluate_gail_validation(self):
        actor_was_training = self.enterprise.actor.training
        discriminator_was_training = self.gail_disc.training
        self.enterprise.actor.eval()
        self.gail_disc.eval()
        with torch.no_grad():
            actor_actions = self.enterprise.actor(self.validation_states)

        action_scale = torch.sqrt(self.act_var + 1e-8)
        expert_actions_n = (self.validation_actions - self.act_mean) / action_scale
        actor_actions_n = (actor_actions - self.act_mean) / action_scale
        random_actions_n = (self.validation_random_actions - self.act_mean) / action_scale

        expert_scores = self._discriminator_scores(
            self.validation_states, expert_actions_n
        )
        actor_scores = self._discriminator_scores(
            self.validation_states, actor_actions_n
        )
        random_scores = self._discriminator_scores(
            self.validation_states, random_actions_n
        )
        if actor_was_training:
            self.enterprise.actor.train()
        if discriminator_was_training:
            self.gail_disc.train()

        actor_auc = self._roc_auc(expert_scores, actor_scores)
        random_auc = self._roc_auc(expert_scores, random_scores)
        expert_mean = float(np.mean(expert_scores))
        actor_mean = float(np.mean(actor_scores))
        random_mean = float(np.mean(random_scores))
        random_shortfall = max(0.0, self.validation_random_auc_min - random_auc)
        selection_score = (
            abs(actor_auc - 0.5)
            + abs(expert_mean - actor_mean)
            + 10.0 * random_shortfall
        )
        return {
            'training_step': int(self.enterprise.pointer),
            'expert_mean': expert_mean,
            'actor_mean': actor_mean,
            'random_mean': random_mean,
            'actor_auc': float(actor_auc),
            'random_auc': float(random_auc),
            'actor_js_distance': self._js_distance(expert_scores, actor_scores),
            'random_js_distance': self._js_distance(expert_scores, random_scores),
            'random_auc_minimum': self.validation_random_auc_min,
            'random_auc_requirement_met': bool(
                random_auc >= self.validation_random_auc_min
            ),
            'selection_score': float(selection_score),
        }

    def maybe_update_gail_validation(self, force=False):
        if self.scope != 'production1' or not hasattr(self, 'gail_disc'):
            return None
        pointer = int(self.enterprise.pointer)
        if not force and pointer < self.next_validation_step:
            return None
        if not force:
            while self.next_validation_step <= pointer:
                self.next_validation_step += self.validation_interval

        metrics = self._evaluate_gail_validation()
        self.gail_validation_history.append(metrics)
        selection_rank = (
            0 if metrics['random_auc_requirement_met'] else 1,
            metrics['selection_score'],
        )
        if selection_rank < self.best_gail_validation_rank:
            self.best_gail_validation_rank = selection_rank
            self.best_gail_validation_score = metrics['selection_score']
            self.best_gail_checkpoint = {
                'metrics': dict(metrics),
                'actor': self._cpu_state_dict(self.enterprise.actor),
                'critic': self._cpu_state_dict(self.enterprise.critic),
                'discriminator': self._cpu_state_dict(self.gail_disc),
            }
        print(
            'GAIL validation: '
            f"step={metrics['training_step']}, "
            f"actor_auc={metrics['actor_auc']:.4f}, "
            f"random_auc={metrics['random_auc']:.4f}, "
            f"score={metrics['selection_score']:.4f}"
        )
        return metrics

    def run_enterprise(self, state, new_ep):
        if self.scope == 'production1':
            # 始终使用预训练时的标准化参数，保证输入分布稳定
            state_np = np.array(state)
            norm_state = (state_np - self.obs_mean.cpu().numpy()) / np.sqrt(self.obs_var.cpu().numpy() + 1e-8)
            norm_state = np.clip(norm_state, -5.0, 5.0)

            # 调用 TD3 的 choose_action (此时 Actor 已是专家水平)
            h_epi, action = self.enterprise.choose_action(None if new_ep else self.epi, norm_state)
            if new_ep:
                self.epi = copy.deepcopy(h_epi)
            return action

        # 否则（如 consumption1），逻辑照旧走 TD3 的探索/决策逻辑
        state = np.array(state)
        h_epi = None if new_ep else self.epi
        h_epi, action = self.enterprise.choose_action(h_epi, state)
        if new_ep:
            self.epi = copy.deepcopy(h_epi)
        return action

    def env_upd(self, state, action, state_, reward, is_train, is_end=False):
        if not is_train: return None

        final_reward = reward

        # --- 针对 production1 的特殊逻辑 ---
        if self.scope == 'production1' and hasattr(self, 'gail_disc'):
            with torch.no_grad():
                # 1. 状态标准化 (对齐预训练)
                s_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
                s_n = torch.clamp((s_t - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8), -5.0, 5.0)

                s_next_t = torch.as_tensor(state_, dtype=torch.float32, device=self.device)
                s_next_n = torch.clamp((s_next_t - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8), -5.0, 5.0)

                # 2. 动作标准化
                a_t = torch.as_tensor(action, dtype=torch.float32, device=self.device)
                a_n = (a_t - self.act_mean) / torch.sqrt(self.act_var + 1e-8)

                # 3. 计算内部奖励 (仅用于 SwanLab 观察，绝不存入经验池)
                logits = self.gail_disc(s_n.unsqueeze(0), a_n.unsqueeze(0))
                score = torch.sigmoid(logits)
                # Keep the logged diagnostic consistent with the clipped training reward.
                r_int = -torch.log((1.0 - score).clamp_min(1e-6)).item()
                if self.config.GAIL_IMITATION_REWARD_CLIP > 0:
                    r_int = min(r_int, self.config.GAIL_IMITATION_REWARD_CLIP)
                self.last_internal_reward = r_int

                # 保持最纯净的环境奖励
                final_reward = reward

            # 必须把标准化后的状态存入经验池！
            s_to_store = s_n.cpu().numpy()
            s_next_to_store = s_next_n.cpu().numpy()
            self._store_recent_policy(s_to_store, action)

            self.epi = self.enterprise.episode_feedback(
                self.epi, s_to_store, action, final_reward, s_next_to_store if is_end else None
            )

        # --- 其他主体 (如 consumption1) 保持原样 ---
        else:
            self.epi = self.enterprise.episode_feedback(
                self.epi, np.array(state), action, final_reward, np.array(state_) if is_end else None
            )

        loss = self.enterprise.learn()
        if self.scope == 'production1':
            self.maybe_update_gail_validation()
        return loss

    def log(self):
        var = self.enterprise.get_var()
        critic_loss , actor_loss = self.enterprise.get_loss()
        # 安全获取内部奖励，如果不是 production1 则返回 0.0
        internal_reward = getattr(self, 'last_internal_reward', 0.0)
        return var, critic_loss, actor_loss, internal_reward

    def get_show(self):
        return self.enterprise.check_show()
