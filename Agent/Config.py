__all__ = ['Config']

import random


class Config:
    def __init__(self, scope: str, action_dim: int, action_bound: float, state_dim: int = 0, reward_gamma: int = 0.95,
                 memory_capacity: int = 80000,
                 learning_rate_actor: float = 3e-4, learning_rate_critic: float = 3e-4,
                 learning_rate_actor_stable: float = 0.00001, learning_rate_critic_stable: float = 0.00002,
                 learning_rate_decay: float = 0.98, learning_rate_decay_time: int = 200,
                 soft_replace_tau: float = 0.01, batch_size: int = 30000, var_init: float = 0.5,
                 var_stable: float = 0.001,
                 var_drop_at: int = 3000, var_stable_at: int = 50000, var_end_at=float('inf'), random_seed: int = None,
                 smooth_noise: float = 0.01, actor_update_delay_times: int = 3,
                 is_critic_double_network: bool = True, is_actor_update_delay: bool = True,
                 is_QNet_smooth_critic: bool = True, is_lstm: bool = True,
                 policy_noise: float = 0.2, discount: float = 0.99, max_hist_len=8, batch_lstm=256, update_every=50,
                 reward_space: int = 1, ra_obs_space: int = 0, beta=0.2, ra_bound: float = 1.0, lra_ra=3e-4,
                 lrc_ra=3e-4, lr_alpha_ra=1e-4,
                 ra_alpha_autotune=True, ra_alpha=0.2, ra_batch_size=256, ra_policy_frequency=2, ra_target_frequency=1,
                 ra_tau=0.005, ra_gamma=0.99
                 ):
        self.scope = scope
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.REWARD_GAMMA = reward_gamma
        self.MEMORY_CAPACITY = memory_capacity
        self.LEARNING_RATE_ACTOR = learning_rate_actor
        self.LEARNING_RATE_CRITIC = learning_rate_critic
        self.LEARNING_RATE_ACTOR_STABLE = learning_rate_actor_stable
        self.LEARNING_RATE_CRITIC_STABLE = learning_rate_critic_stable
        self.LEARNING_RATE_DECAY = learning_rate_decay
        self.LEARNING_RATE_DECAY_TIME = learning_rate_decay_time
        self.SOFT_REPLACE_TAU = soft_replace_tau
        self.BATCH_SIZE = batch_size
        self.VAR_INIT = var_init
        self.VAR_STABLE = var_stable
        self.VAR_DROP_AT = var_drop_at
        self.VAR_STABLE_AT = var_stable_at
        self.VAR_END_AT = var_end_at
        self.SMOOTH_NOISE = smooth_noise
        self.ACTOR_UPDATE_DELAY_TIMES = actor_update_delay_times
        self.IS_ACTOR_UPDATE_DELAY = is_actor_update_delay
        self.IS_CRITIC_DOUBLE_NETWORK = is_critic_double_network
        self.IS_QNET_SMOOTH_CRITIC = is_QNet_smooth_critic
        self.IS_LSTM = is_lstm
        self.MAX_HIST_LEN = max_hist_len
        self.BATCH_SIZE_LSTM = batch_lstm
        self.POLICY_NOISE = policy_noise
        self.DISCOUNT = discount
        self.UPDATE_EVERY = update_every
        if random_seed is None:
            self.random_seed = random.randint(1, 1000)
        else:
            self.random_seed = random_seed

        # reward agent
        self.REWARD_SPACE = reward_space
        self.RA_OBS_SPACE = ra_obs_space
        self.BETA = beta
        self.RA_BOUND = ra_bound
        self.LEARNING_RATE_RA_ACTOR = lra_ra
        self.LEARNING_RATE_RA_CRITIC = lrc_ra
        self.LEARNING_RATE_RA_ALPHA = lr_alpha_ra
        self.RA_ALPHA_AUTOTUNE = ra_alpha_autotune
        self.RA_ALPHA = ra_alpha
        self.ra_batch_size = ra_batch_size
        self.ra_policy_frequency = ra_policy_frequency
        self.ra_target_frequency = ra_target_frequency
        self.ra_tau = ra_tau
        self.ra_gamma = ra_gamma

    def set_state_dim(self, state_num: int):
        self.state_dim = state_num

    def set_scope(self, scope: str):
        self.scope = scope

    def set_learning_rate(self, learning_rate_actor: float = None, learning_rate_critic: float = None):
        if learning_rate_actor is not None:
            self.LEARNING_RATE_ACTOR = learning_rate_actor
        if learning_rate_critic is not None:
            self.LEARNING_RATE_CRITIC = learning_rate_critic

    def set_seed(self, seed):
        self.random_seed = seed

    def __str__(self):
        res = ''
        for key in self.__dict__:
            print(self.__dict__[key])
            res = res + key + ' : ' + str(self.__dict__[key]) + '\n'
        return res
