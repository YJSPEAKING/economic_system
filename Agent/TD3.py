import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
import copy
import os
from collections import OrderedDict
from Agent.Common.ExperienceReplay_TD3 import Experience_Replay as ExpRep
from . import  Config
from Agent.RuningMeanStd import RunningMeanStd
# from Agent.RuningMeanStd import TfRunningMeanStd

tranLock = True
isPercent = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class TrajectorySequenceStore:
    def __init__(self, seq_len, max_episodes=2000):
        self.seq_len = max(1, int(seq_len))
        self.max_episodes = max_episodes
        self.episodes = OrderedDict()

    def record(self, h_epi, state, action):
        if h_epi is None:
            return
        key = int(h_epi)
        if key not in self.episodes:
            self.episodes[key] = []
        else:
            self.episodes.move_to_end(key)
        self.episodes[key].append((
            np.asarray(state, dtype=np.float32).copy(),
            np.asarray(action, dtype=np.float32).copy()
        ))
        while len(self.episodes) > self.max_episodes:
            self.episodes.popitem(last=False)

    def _history(self, seq, end_pos):
        if not seq:
            return None
        end_pos = int(max(0, min(end_pos, len(seq) - 1)))
        start = max(0, end_pos - self.seq_len + 1)
        window = seq[start:end_pos + 1]
        if len(window) < self.seq_len:
            window = [window[0]] * (self.seq_len - len(window)) + window
        states = np.stack([item[0] for item in window], axis=0)
        actions = np.stack([item[1] for item in window], axis=0)
        return states, actions

    def histories_for(self, pick_epi, pick_pos):
        states, actions = [], []
        for epi, pos in zip(np.asarray(pick_epi).reshape(-1), np.asarray(pick_pos).reshape(-1)):
            seq = self.episodes.get(int(epi))
            hist = self._history(seq, pos) if seq is not None else None
            if hist is None:
                return None
            states.append(hist[0])
            actions.append(hist[1])
        return np.stack(states, axis=0), np.stack(actions, axis=0)

    def sample(self, batch_size):
        candidates = [key for key, seq in self.episodes.items() if len(seq) > 0]
        if not candidates:
            return None
        states, actions = [], []
        for _ in range(batch_size):
            key = random.choice(candidates)
            seq = self.episodes[key]
            hist = self._history(seq, random.randrange(len(seq)))
            states.append(hist[0])
            actions.append(hist[1])
        return np.stack(states, axis=0), np.stack(actions, axis=0)


class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, a_bound, max_seq_len=1, use_history=False, hist_hidden=64, nhead=4):
        super(Actor, self).__init__()

        self.use_history = use_history
        self.l1 = nn.Linear(s_dim,128)
        self.l2 = nn.Linear(128,32)
        self.l3 = nn.Linear(32,a_dim)

        self.hist_input = nn.Linear(s_dim, hist_hidden)
        self.hist_pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hist_hidden))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hist_hidden,
            nhead=nhead,
            dim_feedforward=hist_hidden * 2,
            dropout=0.0,
            activation='gelu',
            batch_first=True
        )
        self.hist_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.hist_norm = nn.LayerNorm(hist_hidden)

        self.hl1 = nn.Linear(s_dim + hist_hidden, 128)
        self.hl2 = nn.Linear(128, 32)
        self.hl3 = nn.Linear(32, a_dim)

        self.a_bound = a_bound

    def _history_feature(self, hist_state):
        hist_state = torch.as_tensor(hist_state, dtype=torch.float32, device=device)
        seq_len = hist_state.size(1)
        h = self.hist_input(hist_state) + self.hist_pos_embed[:, :seq_len, :]
        h = self.hist_encoder(h)
        return self.hist_norm(h[:, -1, :])

    def forward(self, state, hist_state=None):
        #3.24
        # state = torch.FloatTensor(state).to(device)
        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        if self.use_history and hist_state is not None:
            hist_feat = self._history_feature(hist_state)
            hs = torch.cat([state, hist_feat], dim=1)
            a = F.tanh(self.hl1(hs))
            a = F.leaky_relu(self.hl2(a))
            return self.a_bound * torch.tanh(self.hl3(a))

        a = F.tanh(self.l1(state))
        a = F.leaky_relu(self.l2(a))
        return self.a_bound * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, s_dim, a_dim, max_seq_len=1, use_history=False, hist_hidden=64, nhead=4):
        super(Critic,self).__init__()
        self.use_history = use_history

        #Q1
        self.l1 = nn.Linear(s_dim+a_dim,128)
        self.l2 = nn.Linear(128,32)
        self.l3 = nn.Linear(32,1)

        #Q2
        self.l4 = nn.Linear(s_dim+a_dim,128)
        self.l5 = nn.Linear(128,32)
        self.l6 = nn.Linear(32,1)

        self.hist_input = nn.Linear(s_dim + a_dim, hist_hidden)
        self.hist_pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hist_hidden))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hist_hidden,
            nhead=nhead,
            dim_feedforward=hist_hidden * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.hist_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.hist_norm = nn.LayerNorm(hist_hidden)

        self.hl1 = nn.Linear(s_dim + a_dim + hist_hidden, 128)
        self.hl2 = nn.Linear(128, 32)
        self.hl3 = nn.Linear(32, 1)
        self.hl4 = nn.Linear(s_dim + a_dim + hist_hidden, 128)
        self.hl5 = nn.Linear(128, 32)
        self.hl6 = nn.Linear(32, 1)

    def _history_feature(self, hist_state, hist_action):
        hist_state = torch.as_tensor(hist_state, dtype=torch.float32, device=device)
        hist_action = torch.as_tensor(hist_action, dtype=torch.float32, device=device)
        hist_sa = torch.cat([hist_state, hist_action], dim=-1)
        seq_len = hist_sa.size(1)
        h = self.hist_input(hist_sa) + self.hist_pos_embed[:, :seq_len, :]
        h = self.hist_encoder(h)
        return self.hist_norm(h[:, -1, :])

    def forward(self,state,action, hist_state=None, hist_action=None):
        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        action = torch.as_tensor(action, dtype=torch.float32, device=device)
        # state = torch.FloatTensor(state).to(device)
        # action = torch.FloatTensor(action).to(device)

        sa = torch.cat([state,action],1)

        if self.use_history and hist_state is not None and hist_action is not None:
            hist_feat = self._history_feature(hist_state, hist_action)
            hsa = torch.cat([sa, hist_feat], dim=1)

            q1 = F.leaky_relu(self.hl1(hsa))
            q1 = F.leaky_relu(self.hl2(q1))
            q1 = self.hl3(q1)

            q2 = F.leaky_relu(self.hl4(hsa))
            q2 = F.leaky_relu(self.hl5(q2))
            q2 = self.hl6(q2)

            return q1,q2

        q1 = F.leaky_relu(self.l1(sa))
        q1 = F.leaky_relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.leaky_relu(self.l4(sa))
        q2 = F.leaky_relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1,q2


    def Q1(self,state,action, hist_state=None, hist_action=None):
        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        action = torch.as_tensor(action, dtype=torch.float32, device=device)
        # state = torch.FloatTensor(state).to(device)
        # action = torch.FloatTensor(action).to(device)

        sa = torch.cat([state,action],1)

        if self.use_history and hist_state is not None and hist_action is not None:
            hist_feat = self._history_feature(hist_state, hist_action)
            hsa = torch.cat([sa, hist_feat], dim=1)
            q1 = F.leaky_relu(self.hl1(hsa))
            q1 = F.leaky_relu(self.hl2(q1))
            q1 = self.hl3(q1)
            return q1

        q1 = F.leaky_relu(self.l1(sa))
        q1 = F.leaky_relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class TD3(object):
    def __init__(self,config:Config):
        self.a_dim = config.action_dim
        self.s_dim = config.state_dim
        self.a_bound = config.action_bound
        self.scope = config.scope
        self.memory = Memory(capacity=config.MEMORY_CAPACITY, dims=2 * self.s_dim + self.a_dim + 1)
        # ==========================================
        # 🚀 新增：动态探测并设置底层 C++ 随机种子
        # ==========================================
        if hasattr(config, 'random_seed'):
            # 情况1：如果 Memory 类自己就有 set_seed 方法（比如它直接继承了 C++ 壳子）
            if hasattr(self.memory, 'set_seed'):
                self.memory.set_seed(config.random_seed)
            # 情况2：如果 Memory 内部包含了一个叫 exp_rep 的 C++ 实例
            elif hasattr(self.memory, 'exp_rep') and hasattr(self.memory.exp_rep, 'set_seed'):
                self.memory.exp_rep.set_seed(config.random_seed)
            # 情况3：如果是个纯 Numpy 数组经验池，上面两个 if 都不会触发，安全跳过。
            # （因为纯 Numpy 经验池的随机性已经被 System.py 顶部的 seed_everything 控制了！）
        # ==========================================
        self.LR_A = config.LEARNING_RATE_ACTOR
        self.LR_C = config.LEARNING_RATE_CRITIC
        self.LR_A_STABLE = config.LEARNING_RATE_ACTOR_STABLE
        self.LR_C_STABLE = config.LEARNING_RATE_CRITIC_STABLE
        self.LR_DECAY = config.LEARNING_RATE_DECAY
        self.LR_DECAY_TIME = config.LEARNING_RATE_DECAY_TIME
        self.GAMMA = config.REWARD_GAMMA
        self.TAU = config.SOFT_REPLACE_TAU
        self.BATCH_SIZE = config.BATCH_SIZE
        self.learn_start_steps = getattr(config, 'LEARN_START_STEPS', self.BATCH_SIZE)
        self.gail_reward_weight = getattr(config, 'GAIL_REWARD_WEIGHT', 2.0)
        self.gail_warmup_steps = max(1, getattr(config, 'GAIL_WARMUP_STEPS', 5000))
        self.disc_update_ratio = max(1, getattr(config, 'DISC_UPDATE_RATIO', 1))
        self.max_hist_len = max(1, int(getattr(config, 'MAX_HIST_LEN', 1)))
        self.use_transformer_critic = (
            bool(getattr(config, 'USE_TRANSFORMER_CRITIC', False))
            and self.max_hist_len > 1
            and self.scope == 'production1'
        )
        self.use_transformer_actor = (
            bool(getattr(config, 'USE_TRANSFORMER_ACTOR', False))
            and self.max_hist_len > 1
            and self.scope == 'production1'
        )
        self.sequence_store = TrajectorySequenceStore(self.max_hist_len)
        if self.scope == 'production1':
            print(
                f"[Transformer] D=True, Critic={self.use_transformer_critic}, "
                f"Actor={self.use_transformer_actor}, hist_len={self.max_hist_len}"
            )
        #self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.pointer = 0
        # self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.a_dim))
        self.episode_temp = {}
        self.show_lar_a = 1
        # TD3参数
        self.is_delay = config.IS_ACTOR_UPDATE_DELAY
        self.is_double = config.IS_CRITIC_DOUBLE_NETWORK
        self.is_smooth = config.IS_QNET_SMOOTH_CRITIC
        self.update_cnt = 0 # 更新次数
        self.rms = RunningMeanStd(epsilon=0.0,shape=self.s_dim)
        self.is_rms=False
        self.tfRms = {}
        self.toshow = {}
        self.policy_noise =config.POLICY_NOISE
        # LSTM参数
        self.critic_loss=0
        self.actor_loss=0
        self.discount = config.DISCOUNT
        if self.is_delay:
            self.policy_target_update_interval = config.ACTOR_UPDATE_DELAY_TIMES # 策略网络更新频率
        else:
            self.policy_target_update_interval = 1

        if self.is_smooth:
            self.eval_noise_scale = config.SMOOTH_NOISE  # 评估动作噪声缩放
        else:
            self.eval_noise_scale = 0.0  # 评估动作噪声缩放

        # Set seed

        self.set_global_seed(config.random_seed)

        self.var_init = config.VAR_INIT
        self.var_stable = config.VAR_STABLE
        self.var_drop_at = config.VAR_DROP_AT
        self.var_stable_at = config.VAR_STABLE_AT
        self.var_end_at = config.VAR_END_AT

        self.var = self.var_init
        self.lr_a = self.LR_A
        self.lr_c = self.LR_C

        # init Actor and Critic network(eval,target)
        self.actor = Actor(
            self.s_dim,
            self.a_dim,
            self.a_bound,
            max_seq_len=self.max_hist_len,
            use_history=self.use_transformer_actor
        ).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)

        self.critic = Critic(
            self.s_dim,
            self.a_dim,
            max_seq_len=self.max_hist_len,
            use_history=self.use_transformer_critic
        ).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)
        # hard_update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(param.data)
            # Actor
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(param.data)

    def _actor_history_tensor(self, temp, state):
        if not self.use_transformer_actor:
            return None

        state = np.asarray(state, dtype=np.float32).reshape(-1)
        states = temp.setdefault('states', [])
        window = states + [state]
        if len(window) < self.max_hist_len:
            window = [window[0]] * (self.max_hist_len - len(window)) + window
        else:
            window = window[-self.max_hist_len:]

        hist_state = np.stack(window, axis=0)[np.newaxis, :, :]
        return torch.as_tensor(hist_state, dtype=torch.float32, device=device)

    def _remember_actor_state(self, temp, state):
        if not self.use_transformer_actor:
            return

        states = temp.setdefault('states', [])
        states.append(np.asarray(state, dtype=np.float32).reshape(-1).copy())
        if len(states) > self.max_hist_len:
            del states[:-self.max_hist_len]

    def choose_action(self, h_epi, state):
        c = np.array(state)[np.newaxis,:]
        # rms 疑似不管用
        self.rms.update(c)
        if self.is_rms:
            state = (state - self.rms.mean)/(self.rms.var + 1e-5)

        self.var = self.var_init
        if self.pointer >self.var_stable_at:
            self.var = self.var_stable
        else:
            delta_step = self.pointer - self.var_drop_at
            if delta_step > 0 :
                self.var = self.var_init +delta_step * (self.var_stable - self.var_init)/(self.var_stable_at - self.var_drop_at)
        if h_epi is None:
            h_epi = self.memory.new_ep()
            self.episode_temp[h_epi] = dict()
        elif h_epi not in self.episode_temp:
            self.episode_temp[h_epi] = dict()
        temp = self.episode_temp[h_epi]
        #
        '''
        to_run = [self.actor(state[np.newaxis,:])]
        list = to_run
        action = list[0][0]

        if isPercent:
            action = action.cpu().detach() + np.random.normal([0 for i in range(self.a_dim)],self.var)
            for i in range(len(action)):
                if action[i] > self.a_bound:
                    action[i] = action[i] % self.a_bound
                if action[i] < -self.a_bound:
                    action[i] = action[i] % -self.a_bound
        else:
            action = np.clip(action + np.random.normal([0 for i in range(self.a_dim)],self.var), 0.001, 100000000) # 固定值

        '''
        state_array = np.asarray(state, dtype=np.float32)
        actor_hist_state = self._actor_history_tensor(temp, state_array)
        list = [self.actor(state_array.reshape(1, -1), actor_hist_state)]
        action = list[0][0]
        if isPercent:
            action = action.cpu().detach() + np.random.normal([0 for i in range(self.a_dim)], self.var)
            for i in range(len(action)):
                if action[i] > self.a_bound:
                    action[i] = action[i] % self.a_bound
                if action[i] < -self.a_bound:
                    action[i] = action[i] % -self.a_bound
        else:
            action = np.clip(action.cpu().detach() + np.random.normal([0 for i in range(self.a_dim)], self.var), 0.001,
                             100000000)  # 固定值

        self._remember_actor_state(temp, state_array)
        return h_epi,action.numpy()

    def episode_feedback(self,h_epi, state, action, reward, final_state):
        episode_finished = final_state is not None
        if final_state is not None:
            final_state = np.zeros(len(final_state))
        self.pointer += 1
        if self.pointer%100 == 0:
            print(self.mark())



        ret_h_epi = self.memory.store_transition(h_epi, state, action, reward, final_state)
        self.sequence_store.record(h_epi, state, action)
        if episode_finished:
            self.episode_temp.pop(h_epi, None)
        return ret_h_epi

    def mark(self):
        return "_______________________________________@(@*#(@#*( " + str(self.pointer) + " " + str(
            self.var) + " " + str(self.show_lar_a) + "_______________________________24$@A#@$"

    def _make_not_done_mask(self, batch, next_state):
        seq_len_next = np.array(batch[5]).reshape(-1, 1)
        not_done = (seq_len_next > 0).astype(np.float32)
        next_state_flat = np.array(next_state).reshape(-1, self.s_dim)
        zero_next_state = np.all(np.isclose(next_state_flat, 0.0), axis=1, keepdims=True)
        not_done[zero_next_state] = 0.0
        return not_done

    def _critic_history_from_batch(self, batch, state, action):
        if not (self.use_transformer_critic or self.use_transformer_actor):
            return None, None

        hist = self.sequence_store.histories_for(batch[6], batch[7])
        if hist is None:
            hist_state = np.repeat(np.asarray(state, dtype=np.float32)[:, np.newaxis, :], self.max_hist_len, axis=1)
            hist_action = np.repeat(np.asarray(action, dtype=np.float32)[:, np.newaxis, :], self.max_hist_len, axis=1)
        else:
            hist_state, hist_action = hist

        hist_state = torch.as_tensor(hist_state, dtype=torch.float32, device=device).detach()
        hist_action = torch.as_tensor(hist_action, dtype=torch.float32, device=device)
        return hist_state, hist_action

    def _next_state_history(self, hist_state, next_state):
        if hist_state is None:
            return None
        next_state = torch.as_tensor(next_state, dtype=torch.float32, device=device)
        return torch.cat([hist_state[:, 1:, :], next_state.unsqueeze(1)], dim=1)

    def _next_action_history(self, hist_action, next_action):
        if hist_action is None:
            return None
        next_action = torch.as_tensor(next_action, dtype=torch.float32, device=device)
        return torch.cat([hist_action[:, 1:, :], next_action.unsqueeze(1)], dim=1)

    def _replace_last_history_action(self, hist_action, action):
        if hist_action is None:
            return None
        return torch.cat([hist_action[:, :-1, :], action.unsqueeze(1)], dim=1)

    def learn(self):

        if self.pointer < self.learn_start_steps:
            return 0
        self.update_cnt += 1
        if self.pointer > 8000 and self.pointer%100 == 0:
            a = 1

        # 从replay buffer（通过memory调用）中随机采样的样本数据
        b_M = self.memory.sample(self.BATCH_SIZE)
        if b_M is None:
            return 0
        # b_M = self.memory.sample(2)
        # 数据处理（归一化） ，b_s_rm,b_s__rm 当前状态和下一个状态的处理后的观测数据
        if self.is_rms:
            b_s_rm = (b_M[0] - self.rms.mean) / (self.rms.var + 1e-5)
            b_s__rm = (b_M[3] - self.rms.mean) / (self.rms.var + 1e-5)
        else:
            b_s_rm = b_M[0]
            b_s__rm = b_M[3]
        b_s = b_s_rm.reshape(-1, self.s_dim)

        # 根据网络设定获取当前状态b_s，动作b_a，奖励b_r，下一个状态b_s_的数据，进行数据处理
        b_a = np.array(b_M[1]).reshape(-1, self.a_dim)
        b_r = b_M[2].reshape(-1, 1)
        b_not_done = self._make_not_done_mask(b_M, b_s__rm)

        # 🛡️ 强行转换为 Tensor 并切断一切图联系 (物理隔离)
        b_s_tensor = torch.as_tensor(b_s, dtype=torch.float32, device=device).detach()
        b_a_tensor = torch.as_tensor(b_a, dtype=torch.float32, device=device)
        b_s_ = torch.as_tensor(b_s__rm.reshape(-1, self.s_dim), dtype=torch.float32, device=device).detach()
        b_r_tensor = torch.as_tensor(b_r, dtype=torch.float32, device=device).detach()
        b_not_done_tensor = torch.as_tensor(b_not_done, dtype=torch.float32, device=device).detach()
        b_s_hist_tensor, b_a_hist_tensor = self._critic_history_from_batch(b_M, b_s, b_a)
        fake_s_n = b_s_tensor
        fake_a_n = b_a_tensor
        if hasattr(self, 'act_mean') and hasattr(self, 'act_var'):
            fake_a_n = (b_a_tensor - self.act_mean) / torch.sqrt(self.act_var + 1e-8)


        # 根据当前训练步数（pointer）计算Critic和Actor网络的学习率lr_c，lr_a
        # 其实没什么用
        train_age = max(0, self.pointer - self.learn_start_steps)
        self.lr_a = max(self.LR_A_STABLE, self.LR_A * np.power(self.LR_DECAY, (train_age / self.LR_DECAY_TIME)))
        self.lr_c = max(self.LR_C_STABLE, self.LR_C * np.power(self.LR_DECAY, (train_age / self.LR_DECAY_TIME)))
        self.show_lar_a = self.lr_a

        if (not tranLock) or self.pointer < self.var_end_at:

            # ==========================================
            # 🚀 阶段三：判别器在线对抗更新 (加强版 - 增加更新步数比)
            # ==========================================
            # 设置步数比 n:1，这里 n=3 代表判别器学3次，Actor/Critic才学1次
            disc_update_ratio = self.disc_update_ratio

            if hasattr(self, 'gail_disc') and hasattr(self, 'sample_expert'):
                # 开启循环“加练”模式
                for _ in range(disc_update_ratio):
                    expert_seq = None
                    fake_seq = None
                    if hasattr(self, 'sample_expert_sequence'):
                        expert_seq = self.sample_expert_sequence(self.BATCH_SIZE, self.max_hist_len)
                        fake_seq = self.sequence_store.sample(self.BATCH_SIZE)
                    if expert_seq is not None and fake_seq is not None:
                        expert_s, expert_a = expert_seq
                        fake_s_seq, fake_a_seq = fake_seq
                        expert_s_n = torch.clamp((expert_s - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8), -5.0,
                                                 5.0)
                        expert_a_n = (expert_a - self.act_mean) / torch.sqrt(self.act_var + 1e-8)
                        fake_s_n_disc = torch.as_tensor(fake_s_seq, dtype=torch.float32, device=device)
                        fake_a_n_disc = torch.as_tensor(fake_a_seq, dtype=torch.float32, device=device)
                        fake_a_n_disc = (fake_a_n_disc - self.act_mean) / torch.sqrt(self.act_var + 1e-8)

                        self.disc_optimizer.zero_grad()
                        real_logits = self.gail_disc(expert_s_n, expert_a_n)
                        fake_logits = self.gail_disc(fake_s_n_disc, fake_a_n_disc)
                        loss_D_real = F.binary_cross_entropy_with_logits(real_logits, torch.full_like(real_logits, 0.9))
                        loss_D_fake = F.binary_cross_entropy_with_logits(fake_logits, torch.full_like(fake_logits, 0.1))
                        loss_D = loss_D_real + loss_D_fake
                        loss_D.backward()
                        self.disc_optimizer.step()
                        continue
                    expert_s, expert_a = self.sample_expert(self.BATCH_SIZE)
                    if expert_s is not None:
                        # 1. 数据标准化 (真/假数据)
                        expert_s_n = torch.clamp((expert_s - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8), -5.0,
                                                 5.0)
                        expert_a_n = (expert_a - self.act_mean) / torch.sqrt(self.act_var + 1e-8)

                        # 注意：这里直接用 self.memory.sample 重新抽样，
                        # 或者为了性能，也可以复用外层 b_s/b_a
                        b_a_tensor = torch.as_tensor(b_a, dtype=torch.float32, device=device)
                        fake_a_n = (b_a_tensor - self.act_mean) / torch.sqrt(self.act_var + 1e-8)
                        fake_s_n = b_s_tensor

                        # 2. 对抗学习
                        self.disc_optimizer.zero_grad()
                        real_logits = self.gail_disc(expert_s_n, expert_a_n)
                        fake_logits = self.gail_disc(fake_s_n, fake_a_n)

                        loss_D_real = F.binary_cross_entropy_with_logits(real_logits, torch.full_like(real_logits, 0.9))
                        loss_D_fake = F.binary_cross_entropy_with_logits(fake_logits, torch.full_like(fake_logits, 0.1))

                        loss_D = loss_D_real + loss_D_fake
                        loss_D.backward()
                        self.disc_optimizer.step()
            # ==========================================

            with torch.no_grad():
                # 计算扰动噪声后的动作 action_with_noise
                next_critic_hist_s = self._next_state_history(b_s_hist_tensor, b_s_)
                target_action = self.actor_target(b_s_, next_critic_hist_s)
                if self.is_smooth:
                    sample = torch.distributions.Normal(0., 1.)
                    a_dim = (self.a_dim,)
                    sample_ = sample.sample(a_dim)
                    noise = torch.clamp(sample_ * self.eval_noise_scale, -2 * self.eval_noise_scale,
                                        2 * self.eval_noise_scale)
                    noise = noise.to(device)
                    action_with_noise = (target_action + noise).clamp(-self.a_bound, self.a_bound)
                else:
                    action_with_noise = target_action.clamp(-self.a_bound, self.a_bound)

                # ==========================================
                # 🚀 核心修复：实时计算动态内部奖励！
                # ==========================================
                # 默认先使用纯环境奖励
                b_r_tensor_fused = b_r_tensor

                # 如果挂载了判别器，用它对刚抽样出的 b_s 和 b_a 进行实时打分
                if hasattr(self, 'gail_disc'):
                    # fake_s_n 和 fake_a_n 在上面的判别器更新块里已经标准化过了
                    reward_seq = self.sequence_store.histories_for(b_M[6], b_M[7])
                    if reward_seq is not None:
                        reward_s_seq, reward_a_seq = reward_seq
                        reward_s_seq = torch.as_tensor(reward_s_seq, dtype=torch.float32, device=device)
                        reward_a_seq = torch.as_tensor(reward_a_seq, dtype=torch.float32, device=device)
                        reward_a_seq = (reward_a_seq - self.act_mean) / torch.sqrt(self.act_var + 1e-8)
                        disc_logits = self.gail_disc(reward_s_seq, reward_a_seq)
                    else:
                        disc_logits = self.gail_disc(fake_s_n, fake_a_n)
                    # 采用 Sigmoid 将分数平滑限制在 0~1 之间，绝对不会造成 Q 值爆炸
                    disc_prob = torch.sigmoid(disc_logits).clamp(1e-4, 1 - 1e-4)
                    raw_r_int = -torch.log1p(-disc_prob)
                    dynamic_r_int = torch.tanh((raw_r_int - math.log(2.0)) / 2.0)

                    # 此时的融合权重 w_gail。建议从 1.0 或 2.0 开始试。
                    gail_scale = min(1.0, train_age / self.gail_warmup_steps)
                    w_gail = self.gail_reward_weight * gail_scale
                    b_r_tensor_fused = b_r_tensor + w_gail * dynamic_r_int
                # ==========================================

                # 计算 target Q 值
                next_critic_hist_a = self._next_action_history(b_a_hist_tensor, action_with_noise)
                target_Q1, target_Q2 = self.critic_target(
                    b_s_, action_with_noise, next_critic_hist_s, next_critic_hist_a
                )
                target_Q = torch.min(target_Q1, target_Q2)

                # 🎯 使用融合了【实时判别器打分】和【真实环境利润】的混合奖励去更新 Critic！
                target_Q = b_r_tensor_fused + self.GAMMA * b_not_done_tensor * target_Q

            # 获得当前 batch 的 Q estimates
            current_Q1, current_Q2 = self.critic(b_s_tensor, b_a_tensor, b_s_hist_tensor, b_a_hist_tensor)
            # 计算 critic loss = td - error
            self.critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            self.critic_loss.backward()
            self.critic_optimizer.step()

            # 延迟策略更新
            if self.update_cnt % self.policy_target_update_interval == 0:
                # ==========================================
                # 🚀 纯粹的 Actor 更新 (Pure Actor)
                # Actor 绝不接触专家数据，仅通过最大化 Critic 的 Q 值来进化！
                # ==========================================
                actor_action = self.actor(b_s_tensor, b_s_hist_tensor)
                actor_hist_a = self._replace_last_history_action(b_a_hist_tensor, actor_action)
                self.actor_loss = -self.critic.Q1(
                    b_s_tensor, actor_action, b_s_hist_tensor, actor_hist_a
                ).mean()

                self.actor_optimizer.zero_grad()
                self.actor_loss.backward()
                self.actor_optimizer.step()

                # soft update
                #   Critic
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)
                #   Actor
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)
        # 只计算critic 的loss 不进行网络更新
        else:
            with torch.no_grad():
                next_critic_hist_s = self._next_state_history(b_s_hist_tensor, b_s_)
                target_action = self.actor_target(b_s_, next_critic_hist_s)
                # 计算扰动噪声后的动作a_ (没有用师兄原本的噪声， 用的td3 的噪声
                if self.is_smooth:
                    # noise = (torch.randn_like(torch.FloatTensor(b_a)) * self.policy_noise).clamp(-self.a_bound,self.a_bound)
                    # action_with_noise = (self.actor_target(b_s_) + noise).clamp(-self.a_bound,self.a_bound)

                    sample = torch.distributions.Normal(0., 1.)
                    a_dim = (self.a_dim,)
                    sample_ = sample.sample(a_dim)
                    noise = torch.clamp(sample_ * self.eval_noise_scale, -2 * self.eval_noise_scale,
                                        2 * self.eval_noise_scale)
                    # 3.24
                    noise = noise.to(device)
                    action_with_noise = (target_action + noise).clamp(-self.a_bound, self.a_bound)
                    # action_with_noise = (self.actor_target(b_s_) + noise).clamp(-self.a_bound, self.a_bound)
                else:
                    action_with_noise = target_action.clamp(-self.a_bound, self.a_bound)

                # 计算target Q 值
                next_critic_hist_a = self._next_action_history(b_a_hist_tensor, action_with_noise)
                target_Q1,target_Q2 = self.critic_target(
                    b_s_, action_with_noise, next_critic_hist_s, next_critic_hist_a
                )
                target_Q = torch.min(target_Q1, target_Q2)
                # target_Q = torch.tensor(b_r) + self.GAMMA * target_Q * self.discount
                #3.24
                target_Q = torch.as_tensor(b_r, dtype=torch.float32, device=device) + self.GAMMA * b_not_done_tensor * target_Q
                # target_Q = torch.tensor(b_r) + self.GAMMA * target_Q
            # 获得当前batch Q estimates
            current_Q1,current_Q2 = self.critic(b_s_tensor, b_a_tensor, b_s_hist_tensor, b_a_hist_tensor)
            # 计算critic loss = td - error
            self.critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        return self.critic_loss

        # 将经验存储到缓冲池（memory）中
        # 参数：时间步数（或者称作episode的索引）、当前状态、选择的动作、获得的奖励以及下一个状态
    def store_transition(self, h_epi, s, a, r, s_):
        # transition = np.hstack((s, a, [r], s_))
        # index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        # self.memory[index, :] = transition
        self.pointer += 1  # 存储了一条新经验
        self.memory.store_transition(h_epi, s, a, r, s_)  # 将当前状态、动作、奖励和下一个状态作为一条完整的经验（transition）存储到经验缓冲池中

    def translate(self,data):
        for d in range(len(data)):
            data[d] = data[d]/1000
        return np.array(data)

    def set_global_seed(self,seed):
        # pytorch_seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # python_seed
        random.seed(seed)

        #NumPy_seed
        np.random.seed(seed)

        # 设置python的hash随机种子
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.deterministic = True
        # 设置PyTorch使用的算法为确定性算法
        torch.backends.cudnn.benchmark = False

    def get_var(self):
        return self.var

    def get_loss(self):
        return self.critic_loss,self.actor_loss




class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0
        self.exp_rep = ExpRep(capacity, 1, False,False,False)
        self.sample_rate=0.03

    def store_transition(self,h_epi, s, a, r, s_):
        #在此处将array转为c_int64
        a = self.exp_rep.encoded_actions(action=a)

        return self.exp_rep.record(h_epi,s,a,r,s_)


    def sample(self, n):
        # assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        # indices = np.random.choice(self.capacity, size=n)
        # return self.data[indices, :]
        return self.exp_rep.get_random_batch(n,self.sample_rate)

    def new_ep(self):
        return self.exp_rep.new_episode()
