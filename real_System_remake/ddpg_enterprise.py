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
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
# import tensorflow as tf
from new_calculate import *
# from Agent.DDPG import DDPG
from Agent.TD3 import TD3
from real_System_remake.pretrain_real_gail import RealActor, normalize, RunningMeanStd

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
    def __init__(self, s_dim=33, a_dim=4, hidden_size=100, max_seq_len=6, nhead=4):
        super(RealDiscriminator, self).__init__()
        self.max_seq_len = max_seq_len
        self.net = nn.Sequential(
            nn.Linear(s_dim + a_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1) # 输出 Logits
        )

        self.seq_input = nn.Linear(s_dim + a_dim, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.seq_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.seq_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        if x.dim() == 3:
            seq_len = x.size(1)
            h = self.seq_input(x) + self.pos_embed[:, :seq_len, :]
            h = self.seq_encoder(h)
            return self.seq_head(h[:, -1, :])
        return self.net(x)

class enterprise_nnu:
    def __init__(self, config: Config):
        self.scope = config.scope
        self.enterprise = TD3(config=config)  # 正常的 TD3 实例
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.scope == 'production1':
            print(f"=== 🚀 {self.scope} 启动端到端联合训练 (DARL) 模式 ===")
            current_dir = os.path.dirname(os.path.abspath(__file__))

            # 保留这句提示，代表它是随机初始化的
            print(f"🌱 TD3 Actor 将从零开始与环境及判别器进行对抗训练")

            # 2. 读取静态标准化参数 (保持原样)
            rms_path = os.path.join(current_dir, 'obs_rms_params.pth')
            rms_params = torch.load(rms_path, map_location=self.device)
            self.obs_mean = rms_params['mean'].to(self.device)
            self.obs_var = rms_params['var'].to(self.device)
            self.act_mean = rms_params['act_mean'].to(self.device)
            self.act_var = rms_params['act_var'].to(self.device)

            # 3. 【阶段一：加载】建立在线专家记忆库 (Expert Buffer)
            csv_path = os.path.join(current_dir, 'expert_data_production1_cleaned.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, header=None)
                expert_data = torch.FloatTensor(df.values).to(self.device)
                # 切分状态与动作 (前33是状态，后4是动作)
                self.expert_states = expert_data[:, :33]
                self.expert_actions = expert_data[:, 33:37]
                self.expert_size = len(self.expert_states)
                marker = pd.to_numeric(df.iloc[:, 0], errors='coerce').round(5).eq(0.1)
                group_id = marker.cumsum()
                self.expert_episodes = []
                for _, group in df.groupby(group_id):
                    values = torch.FloatTensor(group.values).to(self.device)
                    if len(values) > 0:
                        self.expert_episodes.append((values[:, :33], values[:, 33:37]))
                print(f"✅ 专家记忆库已挂载！共包含 {self.expert_size} 条记录。")
            else:
                raise FileNotFoundError(f"❌ 找不到专家数据文件: {csv_path}")

            # 4. 【阶段二：唤醒】加载判别器并解冻
            self.gail_disc = RealDiscriminator(
                s_dim=33,
                a_dim=4,
                max_seq_len=getattr(config, 'MAX_HIST_LEN', 6),
                nhead=getattr(config, 'TRANSFORMER_NHEAD', 4)
            ).to(self.device)
            disc_path = os.path.join(current_dir, 'pretrained_discriminator.pth')
            if os.path.exists(disc_path):
                self.gail_disc.load_state_dict(torch.load(disc_path, map_location=self.device), strict=False)

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
            self.enterprise.sample_expert_sequence = self.sample_expert_sequence
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

    def sample_expert_sequence(self, batch_size, seq_len):
        if not getattr(self, 'expert_episodes', None):
            return None
        states, actions = [], []
        for _ in range(batch_size):
            ep_idx = torch.randint(0, len(self.expert_episodes), (1,), device=self.device).item()
            ep_s, ep_a = self.expert_episodes[ep_idx]
            end = torch.randint(0, len(ep_s), (1,), device=self.device).item()
            start = max(0, end - seq_len + 1)
            s_window = ep_s[start:end + 1]
            a_window = ep_a[start:end + 1]
            if len(s_window) < seq_len:
                pad_len = seq_len - len(s_window)
                s_window = torch.cat([s_window[:1].repeat(pad_len, 1), s_window], dim=0)
                a_window = torch.cat([a_window[:1].repeat(pad_len, 1), a_window], dim=0)
            states.append(s_window)
            actions.append(a_window)
        return torch.stack(states, dim=0), torch.stack(actions, dim=0)

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
                # 建议这里直接用 score.item()，用 -log 如果不稳定会导致数值爆炸
                r_int = score.item()
                self.last_internal_reward = r_int

                # 保持最纯净的环境奖励
                final_reward = reward

            # 必须把标准化后的状态存入经验池！
            s_to_store = s_n.cpu().numpy()
            s_next_to_store = s_next_n.cpu().numpy()

            self.epi = self.enterprise.episode_feedback(
                self.epi, s_to_store, action, final_reward, s_next_to_store if is_end else None
            )

        # --- 其他主体 (如 consumption1) 保持原样 ---
        else:
            self.epi = self.enterprise.episode_feedback(
                self.epi, np.array(state), action, final_reward, np.array(state_) if is_end else None
            )

        loss = self.enterprise.learn()
        return loss

    def log(self):
        var = self.enterprise.get_var()
        critic_loss , actor_loss = self.enterprise.get_loss()
        # 安全获取内部奖励，如果不是 production1 则返回 0.0
        internal_reward = getattr(self, 'last_internal_reward', 0.0)
        return var, critic_loss, actor_loss, internal_reward

    def get_show(self):
        return self.enterprise.check_show()
