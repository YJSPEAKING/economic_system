import os
import sys
import io
import csv
from datetime import datetime

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

ACTOR_CHECKPOINT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frozen_actors")
EXCELLENT_SURVIVAL_DAYS = 90
STABLE_SURVIVAL_WINDOW = 100
STABLE_MIN_GOOD_RATE = 0.8
STABLE_SAVE_MIN_AVG_IMPROVEMENT = 0.5
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
    disc_update_ratio=2,
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
    learning_rate_actor=1e-3,
    learning_rate_critic=2e-3,
    learning_rate_decay=1,
    random_seed=184,
    batch_size=1024,
    memory_capacity=200000,
    learn_start_steps=1024,
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
    reward_profit_weight=1.0,
    reward_credit_weight=0.3,
    reward_survival_weight=0.05,
    reward_unmet_credit_weight=0.1,
    reward_default_weight=1.0,
    reward_smooth_weight=0.1,
    reward_value_scale=100.0
)

enterprise_config = Enterprise_config(
    name='',
    output_name='',
    price=8.0, intention=5.0)

enterprise_add_list = {
    'production1': 'K',
    'consumption1': 'L'
}


def _cpu_state_dict(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


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
        self.survival_history = []
        self.best_candidate_survival = -1
        self.best_stable_avg_survival = -1.0
        self.best_stable_good_rate = 0.0

    def _background_actor_payload(self, kind, episode, survival_days, recent_survival_days=None,
                                  stable_avg_survival=None, stable_good_rate=None):
        if self.Agent.get('consumption1') is None or self.Agent.get('bank1') is None:
            raise RuntimeError("消费企业或银行智能体尚未初始化，无法保存Actor权重。")

        consumption_td3 = self.Agent['consumption1'].enterprise
        bank_td3 = self.Agent['bank1'].bank
        recent_survival_days = list(recent_survival_days or [])
        metadata = {
            "kind": kind,
            "seed": self.seed,
            "episode": episode,
            "epiday": self.epiday,
            "survival_days": survival_days,
            "excellent_survival_days": EXCELLENT_SURVIVAL_DAYS,
            "stable_window": STABLE_SURVIVAL_WINDOW,
            "stable_avg_survival": stable_avg_survival,
            "stable_good_rate": stable_good_rate,
            "stable_min_good_rate": STABLE_MIN_GOOD_RATE,
            "recent_survival_days": recent_survival_days,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "note": "用于人类行为采集时初始化并冻结 consumption1 与 bank1 的Actor；采集时应关闭探索噪声。",
        }
        return {
            "metadata": metadata,
            "actors": {
                "consumption1": {
                    "state_dim": consumption_td3.s_dim,
                    "action_dim": consumption_td3.a_dim,
                    "action_bound": consumption_td3.a_bound,
                    "actor_state_dict": _cpu_state_dict(consumption_td3.actor),
                },
                "bank1": {
                    "state_dim": bank_td3.s_dim,
                    "action_dim": bank_td3.a_dim,
                    "action_bound": bank_td3.a_bound,
                    "actor_state_dict": _cpu_state_dict(bank_td3.actor),
                },
            },
        }

    def _append_actor_checkpoint_index(self, metadata, paths):
        os.makedirs(ACTOR_CHECKPOINT_ROOT, exist_ok=True)
        index_path = os.path.join(ACTOR_CHECKPOINT_ROOT, "checkpoint_index.csv")
        exists = os.path.exists(index_path)
        columns = [
            "created_at",
            "kind",
            "seed",
            "episode",
            "epiday",
            "survival_days",
            "stable_window",
            "stable_avg_survival",
            "stable_good_rate",
            "combined_path",
            "consumption_actor_path",
            "bank_actor_path",
        ]
        with open(index_path, "a", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            if not exists:
                writer.writeheader()
            writer.writerow({
                "created_at": metadata["created_at"],
                "kind": metadata["kind"],
                "seed": metadata["seed"],
                "episode": metadata["episode"],
                "epiday": metadata["epiday"],
                "survival_days": metadata["survival_days"],
                "stable_window": metadata["stable_window"],
                "stable_avg_survival": metadata["stable_avg_survival"],
                "stable_good_rate": metadata["stable_good_rate"],
                "combined_path": paths["combined"],
                "consumption_actor_path": paths["consumption1"],
                "bank_actor_path": paths["bank1"],
            })

    def _save_background_actors(self, kind, episode, survival_days, recent_survival_days=None,
                                stable_avg_survival=None, stable_good_rate=None):
        payload = self._background_actor_payload(
            kind=kind,
            episode=episode,
            survival_days=survival_days,
            recent_survival_days=recent_survival_days,
            stable_avg_survival=stable_avg_survival,
            stable_good_rate=stable_good_rate,
        )
        seed_dir = os.path.join(ACTOR_CHECKPOINT_ROOT, f"seed_{self.seed}")
        os.makedirs(seed_dir, exist_ok=True)

        combined_path = os.path.join(seed_dir, f"seed_{self.seed}_{kind}_background_actors.pth")
        consumption_path = os.path.join(seed_dir, f"seed_{self.seed}_{kind}_consumption1_actor.pth")
        bank_path = os.path.join(seed_dir, f"seed_{self.seed}_{kind}_bank1_actor.pth")

        torch.save(payload, combined_path)
        torch.save(
            {"metadata": payload["metadata"], "agent": "consumption1", **payload["actors"]["consumption1"]},
            consumption_path,
        )
        torch.save(
            {"metadata": payload["metadata"], "agent": "bank1", **payload["actors"]["bank1"]},
            bank_path,
        )

        paths = {
            "combined": combined_path,
            "consumption1": consumption_path,
            "bank1": bank_path,
        }

        if kind == "stable":
            alias_paths = {
                "combined": os.path.join(seed_dir, f"seed_{self.seed}_background_actors.pth"),
                "consumption1": os.path.join(seed_dir, f"seed_{self.seed}_consumption1_actor.pth"),
                "bank1": os.path.join(seed_dir, f"seed_{self.seed}_bank1_actor.pth"),
            }
            torch.save(payload, alias_paths["combined"])
            torch.save(
                {"metadata": payload["metadata"], "agent": "consumption1", **payload["actors"]["consumption1"]},
                alias_paths["consumption1"],
            )
            torch.save(
                {"metadata": payload["metadata"], "agent": "bank1", **payload["actors"]["bank1"]},
                alias_paths["bank1"],
            )
            paths.update(alias_paths)

        self._append_actor_checkpoint_index(payload["metadata"], paths)
        avg_text = "" if stable_avg_survival is None else f"，近{STABLE_SURVIVAL_WINDOW}回合平均{stable_avg_survival:.2f}天"
        rate_text = "" if stable_good_rate is None else f"，达标率{stable_good_rate:.2%}"
        print(f"[OK] 已保存{kind}背景Actor：seed={self.seed}，episode={episode}，存活{survival_days}天{avg_text}{rate_text}")
        print(f"   合并权重：{combined_path}")
        return paths

    def _record_survival_and_maybe_save_actors(self, episode, survival_days):
        self.survival_history.append(int(survival_days))

        if survival_days >= EXCELLENT_SURVIVAL_DAYS and survival_days > self.best_candidate_survival:
            self.best_candidate_survival = survival_days
            self._save_background_actors(
                kind="candidate",
                episode=episode,
                survival_days=survival_days,
                recent_survival_days=self.survival_history[-STABLE_SURVIVAL_WINDOW:],
            )

        if len(self.survival_history) < STABLE_SURVIVAL_WINDOW:
            return

        recent = self.survival_history[-STABLE_SURVIVAL_WINDOW:]
        stable_avg = float(np.mean(recent))
        stable_good_rate = sum(day >= EXCELLENT_SURVIVAL_DAYS for day in recent) / STABLE_SURVIVAL_WINDOW
        if stable_avg < EXCELLENT_SURVIVAL_DAYS or stable_good_rate < STABLE_MIN_GOOD_RATE:
            return

        should_save = self.best_stable_avg_survival < 0
        should_save = should_save or stable_avg >= self.best_stable_avg_survival + STABLE_SAVE_MIN_AVG_IMPROVEMENT
        should_save = should_save or (
            abs(stable_avg - self.best_stable_avg_survival) < 1e-9
            and stable_good_rate > self.best_stable_good_rate
        )
        if not should_save:
            return

        self.best_stable_avg_survival = stable_avg
        self.best_stable_good_rate = stable_good_rate
        self._save_background_actors(
            kind="stable",
            episode=episode,
            survival_days=survival_days,
            recent_survival_days=recent,
            stable_avg_survival=stable_avg,
            stable_good_rate=stable_good_rate,
        )

        if use_wandb:
            wandb.log({
                "优秀背景Actor/近百回合平均存活天数": stable_avg,
                "优秀背景Actor/近百回合90天达标率": stable_good_rate,
                "优秀背景Actor/保存回合": episode,
            })

        self.survival_history = []
        self.best_candidate_survival = -1
        self.best_stable_avg_survival = -1.0
        self.best_stable_good_rate = 0.0

    @staticmethod
    def _cpu_state_dict(module):
        return {
            name: value.detach().cpu().clone()
            for name, value in module.state_dict().items()
        }

    def _background_actor_payload(self, kind, episode, survival_days,
                                  stable_avg_survival=None, stable_good_rate=None):
        consumption_agent = self.Agent.get('consumption1')
        bank_agent = self.Agent.get('bank1')
        consumption_td3 = getattr(consumption_agent, 'enterprise', None)
        bank_td3 = getattr(bank_agent, 'bank', None)
        if consumption_td3 is None or bank_td3 is None:
            return None

        recent_survival_days = self.survival_history[-STABLE_SURVIVAL_WINDOW:]
        return {
            'metadata': {
                'kind': kind,
                'seed': self.seed,
                'episode': episode,
                'epiday': self.epiday,
                'survival_days': survival_days,
                'excellent_survival_days': EXCELLENT_SURVIVAL_DAYS,
                'stable_window': STABLE_SURVIVAL_WINDOW,
                'stable_avg_survival': stable_avg_survival,
                'stable_good_rate': stable_good_rate,
                'stable_min_good_rate': STABLE_MIN_GOOD_RATE,
                'recent_survival_days': list(recent_survival_days),
                'created_at': datetime.now().isoformat(timespec='seconds'),
                'note': 'Use these actors as fixed background agents and disable exploration noise when collecting human data.',
            },
            'actors': {
                'consumption1': {
                    'state_dim': consumption_td3.s_dim,
                    'action_dim': consumption_td3.a_dim,
                    'action_bound': consumption_td3.a_bound,
                    'actor_state_dict': self._cpu_state_dict(consumption_td3.actor),
                },
                'bank1': {
                    'state_dim': bank_td3.s_dim,
                    'action_dim': bank_td3.a_dim,
                    'action_bound': bank_td3.a_bound,
                    'actor_state_dict': self._cpu_state_dict(bank_td3.actor),
                },
            },
        }

    def _append_actor_checkpoint_index(self, payload, combined_path):
        index_path = os.path.join(ACTOR_CHECKPOINT_ROOT, 'checkpoint_index.csv')
        os.makedirs(ACTOR_CHECKPOINT_ROOT, exist_ok=True)
        metadata = payload['metadata']
        row = {
            'created_at': metadata['created_at'],
            'kind': metadata['kind'],
            'seed': metadata['seed'],
            'episode': metadata['episode'],
            'epiday': metadata['epiday'],
            'survival_days': metadata['survival_days'],
            'stable_avg_survival': metadata['stable_avg_survival'],
            'stable_good_rate': metadata['stable_good_rate'],
            'checkpoint_path': combined_path,
        }
        fieldnames = list(row.keys())
        file_exists = os.path.exists(index_path)
        with open(index_path, 'a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _save_background_actors(self, kind, episode, survival_days,
                                stable_avg_survival=None, stable_good_rate=None,
                                write_alias=False):
        payload = self._background_actor_payload(
            kind=kind,
            episode=episode,
            survival_days=survival_days,
            stable_avg_survival=stable_avg_survival,
            stable_good_rate=stable_good_rate,
        )
        if payload is None:
            print('[actor-save] skipped: consumption1 or bank1 is not ready.')
            return

        seed_dir = os.path.join(ACTOR_CHECKPOINT_ROOT, f'seed_{self.seed}')
        os.makedirs(seed_dir, exist_ok=True)

        prefix = f'seed_{self.seed}_{kind}'
        combined_path = os.path.join(seed_dir, f'{prefix}_background_actors.pth')
        consumption_path = os.path.join(seed_dir, f'{prefix}_consumption1_actor.pth')
        bank_path = os.path.join(seed_dir, f'{prefix}_bank1_actor.pth')

        torch.save(payload, combined_path)
        torch.save(payload['actors']['consumption1'], consumption_path)
        torch.save(payload['actors']['bank1'], bank_path)

        if write_alias:
            torch.save(payload, os.path.join(seed_dir, f'seed_{self.seed}_background_actors.pth'))
            torch.save(payload['actors']['consumption1'], os.path.join(seed_dir, f'seed_{self.seed}_consumption1_actor.pth'))
            torch.save(payload['actors']['bank1'], os.path.join(seed_dir, f'seed_{self.seed}_bank1_actor.pth'))

        self._append_actor_checkpoint_index(payload, combined_path)
        print(
            f"[actor-save] {kind} seed={self.seed} episode={episode} "
            f"survival={survival_days} checkpoint={combined_path}"
        )

    def _record_survival_and_maybe_save_actors(self, episode, survival_days):
        self.survival_history.append(int(survival_days))

        if survival_days >= EXCELLENT_SURVIVAL_DAYS and survival_days > self.best_candidate_survival:
            self.best_candidate_survival = survival_days
            self._save_background_actors(
                kind='candidate',
                episode=episode,
                survival_days=survival_days,
            )

        if len(self.survival_history) < STABLE_SURVIVAL_WINDOW:
            return

        recent = self.survival_history[-STABLE_SURVIVAL_WINDOW:]
        stable_avg_survival = float(np.mean(recent))
        stable_good_rate = sum(day >= EXCELLENT_SURVIVAL_DAYS for day in recent) / STABLE_SURVIVAL_WINDOW
        is_stable = (
            stable_avg_survival >= EXCELLENT_SURVIVAL_DAYS
            and stable_good_rate >= STABLE_MIN_GOOD_RATE
        )
        has_improved = (
            self.best_stable_avg_survival < 0
            or stable_avg_survival >= self.best_stable_avg_survival + STABLE_SAVE_MIN_AVG_IMPROVEMENT
            or (
                stable_avg_survival >= self.best_stable_avg_survival
                and stable_good_rate > self.best_stable_good_rate
            )
        )
        if is_stable and has_improved:
            self.best_stable_avg_survival = stable_avg_survival
            self.best_stable_good_rate = stable_good_rate
            self._save_background_actors(
                kind='stable',
                episode=episode,
                survival_days=survival_days,
                stable_avg_survival=stable_avg_survival,
                stable_good_rate=stable_good_rate,
                write_alias=True,
            )

    def run(self):

        for episode in range(10000):
            if self.epiday > 180000 and episode % 100 == 0:
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

            survival_days = int(self.env.day)
            self._record_survival_and_maybe_save_actors(episode, survival_days)

        # self.env.finish()


if __name__ == '__main__':
    seeds_to_run = [184, 291, 83, 739, 512, 117, 894, 652]
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
