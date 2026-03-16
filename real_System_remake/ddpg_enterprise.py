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


class enterprise_nnu:
    def __init__(self, config: Config):
        self.scope = config.scope
        self.enterprise = TD3(config=config)
        self.epi = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 仅针对 production1 进行特殊初始化
        if self.scope == 'production1':
            print(f"=== 🤖 {self.scope} 正在加载 GAIL 决策引擎及标准化参数 ===")

            # 1. 初始化模型
            self.gail_actor = RealActor(s_dim=33, a_dim=4, a_bound=0.5).to(self.device)

            # 2. 加载模型权重
            current_dir = os.path.dirname(os.path.abspath(__file__))
            actor_path = os.path.join(current_dir, 'pretrained_actor.pth')

            if os.path.exists(actor_path):
                self.gail_actor.load_state_dict(torch.load(actor_path, map_location=self.device))
                self.gail_actor.eval()
                print(f"✅ 权重加载成功")

                # 3. 从文件读取预训练时的标准化参数
                rms_path = os.path.join(current_dir, 'obs_rms_params.pth')

                if os.path.exists(rms_path):
                    # 显式指定 map_location 为当前 device
                    rms_params = torch.load(rms_path, map_location=self.device)
                    # 确保 tensor 本身也移动过去
                    self.obs_mean = rms_params['mean'].to(self.device)
                    self.obs_var = rms_params['var'].to(self.device)
                    print(f"✅ {self.scope} 状态标准化参数(RMS)加载成功")
                else:
                    # 备用方案：如果文件不存在，抛出异常或使用默认值（不推荐）
                    print(f"⚠️ 警告: 未找到 {rms_path}，production1 的 GAIL 表现将受严重影响")
                    self.obs_mean = torch.zeros(33).to(self.device)
                    self.obs_var = torch.ones(33).to(self.device)

    def run_enterprise(self, state, new_ep):
        if self.scope == 'production1' and hasattr(self, 'gail_actor'):
            # 关键修改：在创建时直接指定 .to(self.device)
            state_tensor = torch.FloatTensor(np.array(state)).to(self.device).unsqueeze(0)

            with torch.no_grad():
                # 标准化计算，此时 state_tensor, obs_mean, obs_var 都在同一设备上
                norm_state = (state_tensor - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8)
                norm_state = torch.clamp(norm_state, -5.0, 5.0)

                # 模型 forward
                action = self.gail_actor(norm_state).detach().cpu().numpy().flatten()
            return action

        # 否则（如 consumption1），逻辑照旧走 TD3 的探索/决策逻辑
        state = np.array(state)
        h_epi = None if new_ep else self.epi
        h_epi, action = self.enterprise.choose_action(h_epi, state)
        if new_ep:
            self.epi = copy.deepcopy(h_epi)
        return action

    def env_upd(self, state, action, state_, reward, is_train, is_end=False):
        # 如果是 production1，我们只记录数据但不让它更新模型（取决于你的需求）
        if self.scope == 'production1':
            # 如果你依然想把数据存入经验池以备后用，保留下面这行，但去掉 learn()
            # self.epi = self.enterprise.episode_feedback(...)
            return None  # 不调用 self.enterprise.learn()

        # 其他智能体逻辑照旧
        if is_train:
            state_ = np.array(state_)
            self.epi = self.enterprise.episode_feedback(self.epi, state, action, reward, state_ if is_end else None)
            loss = self.enterprise.learn()
            return loss
        return None

    def log(self):
        var = self.enterprise.get_var()
        critic_loss , actor_loss = self.enterprise.get_loss()
        return var,critic_loss,actor_loss

    def get_show(self):
        return self.enterprise.check_show()