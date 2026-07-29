import os
import sys
import io

os.environ["PYTHONUTF8"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from real_System_remake.Environment import Environment
from real_System_remake.Enterprise_config import Enterprise_config
from real_System_remake.Bank_config import Bank_config
from real_System_remake.ddpg_enterprise import enterprise_nnu
from real_System_remake.ddpg_bank import bank_nnu
import real_System_remake.Environment as environment_module
# from real_System_remake.td3_enterprise_lstm  import enterprise_nnu
# from real_System_remake.td3_bank_lstm import bank_nnu
# from real_System_remake.td3_enterprise_rbtree  import enterprise_nnu
# from real_System_remake.ddpg_bank import bank_nnu
# from real_System_remake.td3_enterprise_Newlstm  import enterprise_nnu
# from real_System_remake.td3_bank_Newlstm import bank_nnu
# from real_System_remake.ReLara_enterprise  import enterprise_nnu
# from real_System_remake.ReLara_bank import bank_nnu
from Agent.Config import Config
from real_System_remake.Logger import Logger
# import tensorflow as tf
import copy
import atexit
import random
import gc
import torch
import torch.nn as nn
import swanlab as wandb
import numpy as np
import json


CHECKPOINT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "final_weights")


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 牺牲一点点训练速度，换取绝对的复现性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 在程序第一行就执行！
CURRENT_SEED = 184

use_wandb = True
stable_at = 8000
end_at = 100000
use_rbtree = False
max_episodes = 10000
min_logged_episode = 6000
max_total_sim_days = 180000
# Notice 如果修改lstm的隐藏层节点数量，需要去经验池get batch函数里同步修改
enterprise_ddpg_config = Config(
    scope='',
    action_dim=4,
    action_bound=0.5,
    var_drop_at=1024,
    var_stable_at=stable_at,
    var_end_at=end_at,
    learning_rate_actor=1e-3,
    learning_rate_critic=2e-3,
    learning_rate_decay=1,
    random_seed=184,   # 传入 >= 0 的数，彻底锁死 C++ 的随机性；传入 -1，底层就会根据系统毫秒时间完全随机
    batch_size=1024,
    memory_capacity=200000,
    learn_start_steps=1024,
    gail_reward_weight=2.0,
    gail_warmup_steps=5000,
    disc_update_ratio=1,
    smooth_noise=0.01,
    is_QNet_smooth_critic=True,
    soft_replace_tau=0.01,
    actor_update_delay_times=3,
    policy_noise=0.2,
    max_hist_len=6,
    batch_lstm=500,
    ra_obs_space=37,
    lra_ra=1e-4,
    lrc_ra=2e-4
    # state_dim=,  # 第一次调用时再初始化agent，以便动态适应状态空间
)

bank_ddpg_config = Config(
    scope='',
    action_dim=2,
    action_bound=0.5,
    var_drop_at=1024,
    var_stable_at=stable_at,
    var_end_at=end_at,
    learning_rate_actor=3e-4,
    learning_rate_critic=5e-4,
    learning_rate_decay=1,
    random_seed=184,
    batch_size=1024,
    memory_capacity=200000,
    learn_start_steps=1024,
    critic_grad_clip=1.0,
    actor_grad_clip=1.0,
    target_q_clip=100.0,
    critic_loss_type='huber',
    critic_huber_beta=10.0,
    actor_q_clip=100.0,
    critic_output_bound=100.0,
    critic_output_reg_weight=1e-3,
    smooth_noise=0.01,
    is_QNet_smooth_critic=True,
    soft_replace_tau=0.01,
    actor_update_delay_times=3,
    policy_noise=0.2,
    max_hist_len=6,
    batch_lstm=500,
    ra_obs_space=50,
    lra_ra=1e-4,
    lrc_ra=2e-4
)

def apply_run_seed(seed):
    global CURRENT_SEED
    CURRENT_SEED = int(seed)
    seed_everything(CURRENT_SEED)
    enterprise_ddpg_config.set_seed(CURRENT_SEED)
    bank_ddpg_config.set_seed(CURRENT_SEED)
    environment_module.swanlab_config['enterprise_ddpg_config']['random_seed'] = CURRENT_SEED
    environment_module.swanlab_config['bank_ddpg_config']['random_seed'] = CURRENT_SEED

bank_config = Bank_config(
    name='bank1',
    fund=2000,
    fund_rate=1,
    fund_increase=0.1,
    debt_time=5,
    reward_profit_scale=100.0,
    reward_exposure_scale=1000.0,
    reward_profit_weight=1.0,
    reward_fill_weight=0.2,
    reward_exposure_weight=0.03,
    reward_alive_weight=0.05,
    reward_survival_weight=0.15,
    reward_survival_scale=100.0,
    reward_survival_power=2.0,
    reward_survival_milestone_weight=0.15,
    reward_survival_milestone_day=80.0,
    reward_liquidity_weight=0.1,
    reward_liquidity_scale=1000.0,
    reward_due_shortage_weight=0.35,
    reward_due_shortage_cap=1.5,
    reward_clip=5.0,
    fail_reward=-6.0,
    fail_reward_clip=12.0,
    fail_survival_target_day=90.0,
    fail_survival_shortfall_weight=6.0,
    fail_exposure_weight=0.12
)

enterprise_config = Enterprise_config(
    name='',
    output_name='',
    price=8.0, intention=5.0)

enterprise_add_list = {
    'production1': 'K',
    'consumption1': 'L'
}


class System:
    def __init__(self, seed=None):
        self.seed = CURRENT_SEED if seed is None else int(seed)
        apply_run_seed(self.seed)
        self.env = Environment(name=f"seed_{self.seed}", lim_day=100)
        for key in enterprise_add_list:
            config = copy.deepcopy(enterprise_config)
            config.name = key
            config.output_name = enterprise_add_list[key]
            self.env.add_enterprise_agent(config=config)
        self.env.add_bank(bank_config)
        self.env.add_enterprise_thirdmarket(name='production_thirdMarket', output_name='K', price=100)
        self.env.add_enterprise_thirdmarket(name='consumption_thirdMarket', output_name='L', price=100)

        self.env.init()
        self.epiday = 0
        self.e_execute = self.env.get_enterprise_execute()
        self.b_execute = self.env.get_bank_execute()
        self.execute = self.e_execute + self.b_execute
        self.Agent = {}
        for key in self.execute:
            self.Agent[key] = None

    def save_final_checkpoints(self):
        checkpoint_dir = os.path.join(CHECKPOINT_ROOT, f"seed_{self.seed}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        saved_files = {}

        for agent_name in ("production1", "consumption1"):
            agent = self.Agent.get(agent_name)
            td3_agent = getattr(agent, "enterprise", None)
            if td3_agent is None:
                continue
            actor_path = os.path.join(checkpoint_dir, f"{agent_name}_actor.pth")
            critic_path = os.path.join(checkpoint_dir, f"{agent_name}_critic.pth")
            torch.save(td3_agent.actor.state_dict(), actor_path)
            torch.save(td3_agent.critic.state_dict(), critic_path)
            saved_files[f"{agent_name}_actor"] = actor_path
            saved_files[f"{agent_name}_critic"] = critic_path

        bank_agent = self.Agent.get("bank1")
        td3_bank = getattr(bank_agent, "bank", None)
        if td3_bank is not None:
            actor_path = os.path.join(checkpoint_dir, "bank1_actor.pth")
            critic_path = os.path.join(checkpoint_dir, "bank1_critic.pth")
            torch.save(td3_bank.actor.state_dict(), actor_path)
            torch.save(td3_bank.critic.state_dict(), critic_path)
            saved_files["bank1_actor"] = actor_path
            saved_files["bank1_critic"] = critic_path

        production_agent = self.Agent.get("production1")
        if production_agent is not None and hasattr(production_agent, "gail_disc"):
            disc_path = os.path.join(checkpoint_dir, "discriminator.pth")
            torch.save(production_agent.gail_disc.state_dict(), disc_path)
            saved_files["discriminator"] = disc_path

        if production_agent is not None and hasattr(production_agent, "expert_split_metadata"):
            split_path = os.path.join(checkpoint_dir, "expert_split.json")
            with open(split_path, "w", encoding="utf-8") as f:
                json.dump(
                    _json_safe(production_agent.expert_split_metadata),
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            saved_files["expert_split"] = split_path

        obs_keys = ("obs_mean", "obs_var", "act_mean", "act_var")
        if production_agent is not None and all(hasattr(production_agent, key) for key in obs_keys):
            obs_path = os.path.join(checkpoint_dir, "obs_rms_params.pth")
            torch.save(
                {
                    "mean": production_agent.obs_mean.detach().cpu(),
                    "var": production_agent.obs_var.detach().cpu(),
                    "act_mean": production_agent.act_mean.detach().cpu(),
                    "act_var": production_agent.act_var.detach().cpu(),
                },
                obs_path,
            )
            saved_files["obs_rms_params"] = obs_path

        config_path = os.path.join(checkpoint_dir, "config.json")
        saved_files["config"] = config_path
        config_payload = {
            "seed": self.seed,
            "episode": getattr(self.env, "episode", None),
            "total_sim_days": self.epiday,
            "max_episodes": max_episodes,
            "min_logged_episode": min_logged_episode,
            "max_total_sim_days": max_total_sim_days,
            "enterprise_ddpg_config": enterprise_ddpg_config.__dict__,
            "bank_ddpg_config": bank_ddpg_config.__dict__,
            "enterprise_config": enterprise_config.__dict__,
            "bank_config": bank_config.__dict__,
            "swanlab_config": environment_module.swanlab_config,
            "expert_split": getattr(production_agent, "expert_split_metadata", None),
            "saved_files": saved_files,
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(config_payload), f, ensure_ascii=False, indent=2)
        print(f"[Checkpoint] saved final weights for seed {self.seed}: {checkpoint_dir}")
        return checkpoint_dir

    def run(self):

        for episode in range(max_episodes):
            if episode > min_logged_episode and self.epiday > max_total_sim_days and episode % 100 == 0:
                break
            state = self.env.reset()
            last_state = None
            last_action = None
            last_reward_pro = None
            # 第一次进来的话，构建enterprise智能体集合
            for target_key in self.e_execute:
                if self.Agent[target_key] is None:
                    config = copy.deepcopy(enterprise_ddpg_config)
                    config.set_scope(target_key)
                    config.set_state_dim(len(state[target_key]))
                    self.Agent[target_key] = enterprise_nnu(config)
            # 第一次进来的话，构建bank智能体集合
            for target_key in self.b_execute:
                if self.Agent[target_key] is None:
                    config = copy.deepcopy(bank_ddpg_config)
                    config.set_scope(target_key)
                    config.set_state_dim(len(state[target_key]))
                    self.Agent[target_key] = bank_nnu(config)
            new_ep = True
            while True:
                action = {}
                reward_pro = {}
                # TD3
                for target_key in self.e_execute:
                    action[target_key] = self.Agent[target_key].run_enterprise(state[target_key], new_ep)
                for target_key in self.b_execute:
                    action[target_key] = self.Agent[target_key].run_bank(state[target_key], new_ep)

                new_ep = False
                self.epiday = self.epiday + 1

                self.env.step(action)
                next_state, reward, done = self.env.observe()

                # done的情况下，因为已知state 和 state_，reward为破产惩罚，处理逻辑不需要时序错峰
                if done:
                    next_action = {}
                    for target_key in self.e_execute:
                        # next_action[target_key] , reward_pro[target_key] = self.Agent[target_key].run_enterprise(next_state[target_key], new_ep)
                        self.Agent[target_key].env_upd(state=state[target_key],
                                                       action=action[target_key],
                                                       state_=next_state[target_key],
                                                       # reward=reward[target_key]['economy'],
                                                       reward=reward[target_key]['business'],
                                                       # reward=reward[target_key]['eval_business'],
                                                       # reward=reward[target_key]['days'],
                                                       is_train=True,
                                                       # action_ = next_action[target_key],
                                                       # reward_pro = reward_pro[target_key],
                                                       is_end=done
                                                       )
                        # 更新银行feedback 其实和上面一样，但是为了方便以后可能要拓展先区分开来
                    for target_key in self.b_execute:
                        # next_action[target_key], reward_pro[target_key] = self.Agent[target_key].run_bank(next_state[target_key], new_ep)

                        self.Agent[target_key].env_upd(state=state[target_key],
                                                       action=action[target_key],
                                                       state_=next_state[target_key],
                                                       reward=reward[target_key]['WNDB'],
                                                       is_train=True,
                                                       # action_ = next_action[target_key],
                                                       # reward_pro = reward_pro[target_key],
                                                       is_end=done
                                                       )
                    break
                else:
                    # 关键一步 时序错峰，详见时序错峰.png
                    if last_state is not None:
                        # 更新企业feedback
                        for target_key in self.e_execute:
                            self.Agent[target_key].env_upd(state=last_state[target_key],
                                                           action=last_action[target_key],
                                                           state_=state[target_key],
                                                           # reward=reward[target_key]['economy'],
                                                           reward=reward[target_key]['business'],
                                                           # reward=reward[target_key]['days'],
                                                           # reward=reward[target_key]['eval_business'],
                                                           is_train=True,
                                                           # action_ = action[target_key],
                                                           # reward_pro = last_reward_pro[target_key],
                                                           is_end=done
                                                           )
                        # 更新银行feedback 其实和上面一样，但是为了方便以后可能要拓展先区分开来
                        for target_key in self.b_execute:
                            self.Agent[target_key].env_upd(state=last_state[target_key],
                                                           action=last_action[target_key],
                                                           state_=state[target_key],
                                                           reward=reward[target_key]['WNDB'],
                                                           is_train=True,
                                                           # action_=action[target_key],
                                                           # reward_pro=last_reward_pro[target_key],
                                                           is_end=done
                                                           )

                if use_wandb:
                    # 1. bank1 没有被修改，依然用 3 个变量接收
                    var, critic_bank, actor_bank = self.Agent['bank1'].log()

                    # 2. production1 和 consumption1 是 enterprise_nnu，现在会返回 4 个值
                    _, critic_production1, actor_production1, int_r_pro1 = self.Agent['production1'].log()
                    _, crtic_consumption1, actor_consumption1, _ = self.Agent['consumption1'].log()

                    # --- 下面的 log 记录代码保持你刚才的样子不变 ---
                    wandb.log({'actor_loss/bank1': actor_bank})
                    wandb.log({'actor_loss/production1': actor_production1})
                    wandb.log({'actor_loss/consumption1': actor_consumption1})

                    wandb.log({'critic_loss/bank1': critic_bank})
                    wandb.log({'critic_loss/production1': critic_production1})
                    wandb.log({'critic_loss/consumption1': crtic_consumption1})

                    wandb.log({'探索噪声var': var})

                    if 'production1' in self.Agent:
                        wandb.log({'GAIL_Internal_Reward/pro1': int_r_pro1})

                # for target_key in self.e_execute:
                #     print('after_'+target_key+'ra_action', reward_pro[target_key])
                # for target_key in self.b_execute:
                #     print('after_bank_ra_action', reward_pro[target_key])
                last_state = state
                state = next_state
                last_action = action
                last_reward_pro = reward_pro

        self.save_final_checkpoints()
        # self.env.finish()


if __name__ == '__main__':
    seeds_to_run = [184, 291, 83, 739, 894, 652, 187, 191]
    for seed in seeds_to_run:
        system = System(seed=seed)
        system.run()

        if hasattr(system, 'env'):
            system.env.finish()
        del system
        gc.collect()
        torch.nn.Module.dump_patches = True
        torch.cuda.empty_cache()

    # tf.reset_default_graph()
