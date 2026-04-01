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
        self.scope = config.scope
        self.enterprise = TD3(config=config)  # 正常的 TD3 实例
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.scope == 'production1':
            print(f"=== 🤖 {self.scope} 切换至进化型 GAIL-AC 模式 ===")
            current_dir = os.path.dirname(os.path.abspath(__file__))

            # 【A. 权重注入】将预训练权重同步给 TD3 的 Actor，实现热启动
            actor_path = os.path.join(current_dir, 'pretrained_actor.pth')
            if os.path.exists(actor_path):
                pretrained_dict = torch.load(actor_path, map_location=self.device)
                self.enterprise.actor.load_state_dict(pretrained_dict)
                self.enterprise.actor_target.load_state_dict(pretrained_dict)
                print(f"✅ TD3 Actor 已继承专家经验，不再是随机初始化")

            # 【B. 导师加载】加载预训练判别器作为奖励引擎
            self.gail_disc = RealDiscriminator(s_dim=33, a_dim=4).to(self.device)
            disc_path = os.path.join(current_dir, 'pretrained_discriminator.pth')
            if os.path.exists(disc_path):
                self.gail_disc.load_state_dict(torch.load(disc_path, map_location=self.device))
                self.gail_disc.eval()  # 判别器只看不练
                for param in self.gail_disc.parameters():
                    param.requires_grad = False
                print(f"✅ 判别器已挂载，将实时评估动作逻辑")

            # 读取静态标准化参数
            rms_path = os.path.join(current_dir, 'obs_rms_params.pth')
            rms_params = torch.load(rms_path, map_location=self.device)
            self.obs_mean = rms_params['mean'].to(self.device)
            self.obs_var = rms_params['var'].to(self.device)
            # 【新增】加载动作的标准化参数
            self.act_mean = rms_params['act_mean'].to(self.device)
            self.act_var = rms_params['act_var'].to(self.device)

            self.last_r_int = 0.0  # 用于日志

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

                # 3. 计算内部奖励
                logits = self.gail_disc(s_n.unsqueeze(0), a_n.unsqueeze(0))
                score = torch.sigmoid(logits)
                r_int = -torch.log(1 - score + 1e-8).item()

                # 可以在此处通过参数接收 w_gail
                w_gail = 1
                final_reward = reward + w_gail * r_int
                self.last_internal_reward = r_int

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
        return var,critic_loss,actor_loss

    def get_show(self):
        return self.enterprise.check_show()