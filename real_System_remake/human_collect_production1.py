import csv
import os
import random
import sys
import time
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from real_System_remake.Bank_config import Bank_config
from real_System_remake.Enterprise_config import Enterprise_config
from real_System_remake.Environment import Environment


TARGET_AGENT = "production1"
DEFAULT_SEED = 184
DEFAULT_AUTO_POLICY = "td3"
BACKGROUND_ACTOR_SEED = 184
BACKGROUND_ACTOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frozen_actors")
BACKGROUND_ACTOR_PATH = os.environ.get(
    "HUMAN_COLLECT_BACKGROUND_ACTOR_PATH",
    os.path.join(
        BACKGROUND_ACTOR_DIR,
        f"seed_{BACKGROUND_ACTOR_SEED}",
        f"seed_{BACKGROUND_ACTOR_SEED}_background_actors.pth",
    ),
)
ACTION_NAMES = ["WNDF", "K", "L", "price"]
ACTION_DISPLAY_NAMES = ["申请贷款金额", "原料A采购需求", "原料B采购需求", "产品A销售价格"]
ENTERPRISE_ADD_LIST = {
    "production1": "K",
    "consumption1": "L",
}
FIXED_ENTERPRISE_ACTION = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
FIXED_BANK_ACTION = np.array([1.0, 1.0], dtype=float)
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "expert_data_production1_human.csv",
)
DEFAULT_META_OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "expert_data_production1_human_with_meta.csv",
)
DEFAULT_LOG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "human_collect_logs",
)

STATE_FIELDS = [
    ("现金", 1000),
    ("存货", 100),
    ("欠款总额", 1000),
    ("上一天售出数量", 10),
    ("上一天产出数量", 10),
    ("上一天销售价格", 10),
    ("今天预设销售价格", 10),
    ("上一天贷款意愿", 1000),
    ("上一天实际获得贷款", 1000),
    ("今天待还本金", 1000),
    ("今天待还利息", 10),
    ("上一天A采购需求", 10),
    ("上一天B采购需求", 10),
    ("上一天甲公司向甲公司购买价格", 10),
    ("上一天甲公司向甲公司购买数量", 10),
    ("上一天甲公司向乙公司购买价格", 10),
    ("上一天甲公司向乙公司购买数量", 10),
    ("上一天甲公司向第三方市场购买A价格", 10),
    ("上一天甲公司向第三方市场购买A数量", 10),
    ("上一天甲公司向第三方市场购买B价格", 10),
    ("上一天甲公司向第三方市场购买B数量", 10),
    ("上一天乙公司向甲公司购买价格", 10),
    ("上一天乙公司向甲公司购买数量", 10),
    ("上一天乙公司向乙公司购买价格", 10),
    ("上一天乙公司向乙公司购买数量", 10),
    ("上一天乙公司向第三方市场购买A价格", 10),
    ("上一天乙公司向第三方市场购买A数量", 10),
    ("上一天乙公司向第三方市场购买B价格", 10),
    ("上一天乙公司向第三方市场购买B数量", 10),
    ("今天产品A报价：甲公司", 10),
    ("今天产品A报价：第三方市场", 10),
    ("今天产品B报价：乙公司", 10),
    ("今天产品B报价：第三方市场", 10),
]


def state_label(index):
    if index < len(STATE_FIELDS):
        return STATE_FIELDS[index][0]
    return f"未命名状态 {index + 1}"


def display_state_value(index, value):
    scale = STATE_FIELDS[index][1] if index < len(STATE_FIELDS) else 1
    return value * scale


def raw_state_value(state, index):
    return display_state_value(index, state[index])


def format_number(value):
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    if abs(value) >= 1000:
        return f"{value:,.2f}"
    return f"{value:.4g}"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ModuleNotFoundError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_enterprise_config(name, output_name):
    return Enterprise_config(
        name=name,
        output_name=output_name,
        price=8.0,
        intention=5.0,
    )


def make_bank_config():
    return Bank_config(
        name="bank1",
        fund=2000,
        fund_rate=1,
        fund_increase=0.1,
        debt_time=5,
    )


def make_ddpg_config(scope, action_dim, action_bound, state_dim, seed):
    from Agent.Config import Config

    return Config(
        scope=scope,
        action_dim=action_dim,
        action_bound=action_bound,
        state_dim=state_dim,
        var_drop_at=1024,
        var_stable_at=8000,
        var_end_at=100000,
        learning_rate_actor=1e-3,
        learning_rate_critic=2e-3,
        learning_rate_decay=1,
        random_seed=seed,
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
    )


def aggregate_purchase(state, start):
    prices = [raw_state_value(state, start), raw_state_value(state, start + 2),
              raw_state_value(state, start + 4), raw_state_value(state, start + 6)]
    nums = [raw_state_value(state, start + 1), raw_state_value(state, start + 3),
            raw_state_value(state, start + 5), raw_state_value(state, start + 7)]
    total_num = sum(nums)
    if total_num <= 0:
        return 0.0, 0.0
    avg_price = sum(price * num for price, num in zip(prices, nums)) / total_num
    return total_num, avg_price


def market_min(values):
    positives = [value for value in values if value > 0]
    return min(positives) if positives else 0.0


def weighted_average(pairs):
    total_num = sum(num for _, num in pairs)
    if total_num <= 0:
        return 0.0
    return sum(price * num for price, num in pairs) / total_num


def purchase_summary(local_pair, third_pair):
    local_price, local_num = local_pair
    third_price, third_num = third_pair
    total_num = local_num + third_num
    total_spend = local_price * local_num + third_price * third_num
    return (
        f"普通市场：买到 {format_number(local_num)}，单价 {format_number(local_price)}\n"
        f"第三方市场：买到 {format_number(third_num)}，定价 {format_number(third_price)}\n"
        f"合计 {format_number(total_num)}，花费 {format_number(total_spend)}"
    )


def market_price_summary(local_price, third_price):
    cheaper = "普通市场" if local_price <= third_price else "第三方市场"
    return (
        f"普通市场单价 {format_number(local_price)}；第三方市场固定定价 {format_number(third_price)}。\n"
        f"系统会先买更便宜的{cheaper}，买不够再买下一档。"
    )


def display_rows(state, day=None):
    is_first_day = day == 1
    previous_text = "开局默认运行" if is_first_day else "上一天"
    k_local_pair = (raw_state_value(state, 13), raw_state_value(state, 14))
    l_local_pair = (raw_state_value(state, 15), raw_state_value(state, 16))
    k_third_pair = (raw_state_value(state, 17), raw_state_value(state, 18))
    l_third_pair = (raw_state_value(state, 19), raw_state_value(state, 20))
    k_pairs = [k_local_pair, k_third_pair]
    l_pairs = [l_local_pair, l_third_pair]
    previous_purchase_spend = sum(price * num for price, num in k_pairs + l_pairs)
    k_local_price = raw_state_value(state, 29)
    k_third_price = raw_state_value(state, 30)
    l_local_price = raw_state_value(state, 31)
    l_third_price = raw_state_value(state, 32)
    cash = raw_state_value(state, 0)
    payback = raw_state_value(state, 9)
    interest = raw_state_value(state, 10)
    debt_due = payback + interest
    k_need = raw_state_value(state, 11)
    l_need = raw_state_value(state, 12)
    price = raw_state_value(state, 6)
    product_stock = raw_state_value(state, 1)
    previous_sales = raw_state_value(state, 3)
    previous_output = raw_state_value(state, 4)
    previous_price = raw_state_value(state, 5)
    previous_revenue = previous_sales * previous_price
    previous_net = previous_revenue - previous_purchase_spend
    return [
        ("总体", "当前现金（也是申请贷款金额上限）", cash, "cash"),
        ("总体", "今天需要还款（本金+利息）", debt_due, "debt_due"),
        ("总体", "目前总欠款", raw_state_value(state, 2), "debt"),
        ("总体", f"{previous_text}经营差额：收入 {format_number(previous_revenue)} - 原料采购支出 {format_number(previous_purchase_spend)}", previous_net, "previous_net"),
        ("原料A", "原料A作用：和原料B配套投入生产，少的一种会卡住产品A产量", "产品A产量 = 2.5 × min(买到的原料A, 买到的原料B)\n原料当天购买、当天投入生产，不跨天保存。", "production_rule"),
        ("原料A", "当前原料A参考采购量（你可在下方修改）", k_need, "k_need"),
        ("原料A", "今天原料A可购买价格", market_price_summary(k_local_price, k_third_price), "k_market_detail"),
        ("原料A", f"{previous_text}原料A成交结果", purchase_summary(k_local_pair, k_third_pair), "previous_k_trade"),
        ("原料B", "原料B作用：和原料A配套投入生产，少的一种会卡住产品A产量", "产品A产量 = 2.5 × min(买到的原料A, 买到的原料B)\n原料当天购买、当天投入生产，不跨天保存。", "production_rule"),
        ("原料B", "当前原料B参考采购量（你可在下方修改）", l_need, "l_need"),
        ("原料B", "今天原料B可购买价格", market_price_summary(l_local_price, l_third_price), "l_market_detail"),
        ("原料B", f"{previous_text}原料B成交结果", purchase_summary(l_local_pair, l_third_pair), "previous_l_trade"),
        ("产品A", "当前可出售的产品A库存", product_stock, "product_stock"),
        ("产品A", "今天产品A销售价格", price, "price"),
        ("产品A", f"{previous_text}产品A表现：售出数量（产出数量 {format_number(previous_output)}）", previous_sales, "sales"),
    ]


def row_change_tag(key, value, previous_state):
    if previous_state is None:
        return ()
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return ()
    previous = {row[3]: row[2] for row in display_rows(previous_state)}
    if key not in previous:
        return ()
    old = previous[key]
    if not isinstance(old, (int, float, np.integer, np.floating)):
        return ()
    if abs(value - old) < 1e-9:
        return ()
    good_when_up = {"cash", "previous_net", "previous_revenue", "sales", "output", "sales_output"}
    bad_when_up = {"debt", "payback", "interest", "debt_due", "purchase_spend", "k_market", "l_market"}
    if key in good_when_up:
        return ("change_good",) if value > old else ("change_bad",)
    if key in bad_when_up:
        return ("change_bad",) if value > old else ("change_good",)
    return ()


def risk_tag(key, value, state):
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return ()
    cash = raw_state_value(state, 0)
    if key == "cash":
        if value < 50:
            return ("risk_high",)
        if value < 200:
            return ("risk_medium",)
    if key == "debt":
        if cash > 0 and value > cash * 3:
            return ("risk_high",)
        if cash > 0 and value > cash:
            return ("risk_medium",)
    if key in {"payback", "interest", "debt_due"}:
        if cash > 0 and value > cash:
            return ("risk_high",)
        if cash > 0 and value > cash * 0.5:
            return ("risk_medium",)
    return ()


class HumanProductionCollector:
    def __init__(
        self,
        seed=DEFAULT_SEED,
        output_path=DEFAULT_OUTPUT,
        meta_output_path=DEFAULT_META_OUTPUT,
        auto_policy=DEFAULT_AUTO_POLICY,
        participant_id="anonymous",
    ):
        self.seed = int(seed)
        self.output_path = output_path
        self.meta_output_path = meta_output_path
        self.auto_policy = auto_policy
        self.participant_id = participant_id.strip() or "anonymous"
        self.env = None
        self.state = None
        self.new_ep = True
        self.auto_agents = {}
        self.rows_saved = 0
        self.history = []
        self.state_history = []
        self.background_actor_path = BACKGROUND_ACTOR_PATH
        self.background_actor_metadata = None
        self.background_actor_loaded = set()
        seed_everything(self.seed)
        self._build_env()

    def _build_env(self):
        self.env = Environment(
            name=f"human_collect_seed_{self.seed}",
            lim_day=100,
            logger_path=DEFAULT_LOG_DIR,
            use_swanlab=False,
        )
        for key, output_name in ENTERPRISE_ADD_LIST.items():
            self.env.add_enterprise_agent(make_enterprise_config(key, output_name))
        self.env.add_bank(make_bank_config())
        self.env.add_enterprise_thirdmarket("production_thirdMarket", "K", 100)
        self.env.add_enterprise_thirdmarket("consumption_thirdMarket", "L", 100)
        self.env.init()

    def start_episode(self):
        self.state = self.env.reset()
        self.new_ep = True
        self.auto_agents = {}
        self.background_actor_loaded = set()
        self.history = []
        self.state_history = [self._state_snapshot()]
        return self.current_state()

    def _state_snapshot(self):
        return {
            "day": int(self.env.day),
            "full_state": {
                key: np.array(value, dtype=float).copy()
                for key, value in self.state.items()
            },
        }

    def current_state(self):
        return np.array(self.state[TARGET_AGENT], dtype=float)

    def _get_auto_agent(self, key):
        if self.auto_policy == "fixed":
            return None
        if key in self.auto_agents:
            return self.auto_agents[key]
        if key in self.env.get_enterprise_execute():
            from real_System_remake.ddpg_enterprise import enterprise_nnu

            config = make_ddpg_config(key, 4, 0.5, len(self.state[key]), self.seed)
            agent = enterprise_nnu(config)
        else:
            from real_System_remake.ddpg_bank import bank_nnu

            config = make_ddpg_config(key, 2, 0.5, len(self.state[key]), self.seed)
            agent = bank_nnu(config)
        self._load_frozen_background_actor(key, agent)
        self.auto_agents[key] = agent
        return agent

    def _load_frozen_background_actor(self, key, agent):
        if key not in {"consumption1", "bank1"}:
            return
        if not os.path.exists(self.background_actor_path):
            raise FileNotFoundError(
                "找不到成熟乙公司/银行权重文件："
                f"{self.background_actor_path}。请确认 frozen_actors 已复制到 real_System_remake 下。"
            )

        from real_System_remake.frozen_actor_utils import load_background_actors

        if key == "consumption1":
            checkpoint = load_background_actors(
                self.background_actor_path,
                consumption_agent=agent,
                freeze=True,
            )
        else:
            checkpoint = load_background_actors(
                self.background_actor_path,
                bank_agent=agent,
                freeze=True,
            )
        self.background_actor_metadata = checkpoint.get("metadata", {})
        self.background_actor_loaded.add(key)

    def _build_action(self, target_action):
        action = {TARGET_AGENT: np.array(target_action, dtype=float)}

        for key in self.env.get_enterprise_execute():
            if key == TARGET_AGENT:
                continue
            if self.auto_policy == "fixed":
                action[key] = FIXED_ENTERPRISE_ACTION.copy()
            else:
                action[key] = self._get_auto_agent(key).run_enterprise(self.state[key], self.new_ep)

        for key in self.env.get_bank_execute():
            if self.auto_policy == "fixed":
                action[key] = FIXED_BANK_ACTION.copy()
            else:
                action[key] = self._get_auto_agent(key).run_bank(self.state[key], self.new_ep)
        return action

    def _auto_target_action(self):
        if self.auto_policy == "fixed":
            return FIXED_ENTERPRISE_ACTION.copy()
        return self._get_auto_agent(TARGET_AGENT).run_enterprise(self.state[TARGET_AGENT], self.new_ep)

    def step(self, model_action, human_values, decision_seconds):
        model_action = np.array(model_action, dtype=float)
        if model_action.shape[0] != len(ACTION_NAMES):
            raise ValueError("甲公司动作必须是4个数。")

        action = self._build_action(model_action)
        state_before_action = self.current_state().copy()
        full_state_before_action = {
            key: np.array(value, dtype=float).copy()
            for key, value in self.state.items()
        }
        day_before_action = self.env.day

        self.history.append(
            {
                "day": day_before_action,
                "state": state_before_action,
                "full_state": full_state_before_action,
                "action": model_action.copy(),
                "human_values": list(human_values),
            }
        )
        self._save_row(state_before_action, model_action, human_values, day_before_action, decision_seconds)
        self.env.step(action)
        next_state, reward, done = self.env.observe()
        self.state = next_state
        self.new_ep = False
        self.state_history.append(self._state_snapshot())
        return done, reward

    def auto_step(self):
        state_before_action = self.current_state().copy()
        full_state_before_action = {
            key: np.array(value, dtype=float).copy()
            for key, value in self.state.items()
        }
        day_before_action = self.env.day
        target_action = self._auto_target_action()
        action = self._build_action(target_action)
        self.env.step(action)
        next_state, reward, done = self.env.observe()
        self.state = next_state
        self.new_ep = False
        self.state_history.append(self._state_snapshot())
        return {
            "done": done,
            "reward": reward,
            "day": day_before_action,
            "state": state_before_action,
            "full_state": full_state_before_action,
            "action": np.array(target_action, dtype=float),
        }

    def _save_row(self, state, model_action, human_values, day, decision_seconds):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow([*state.tolist(), *model_action.tolist()])

        meta_exists = os.path.exists(self.meta_output_path)
        with open(self.meta_output_path, "a", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            if not meta_exists:
                writer.writerow(
                    [
                        "participant_id",
                        "seed",
                        "episode",
                        "day",
                        "auto_policy",
                        "decision_seconds",
                        *[f"state_{i}_{state_label(i)}" for i in range(len(state))],
                        *[f"human_{name}" for name in ACTION_DISPLAY_NAMES],
                        *[f"model_action_{name}" for name in ACTION_NAMES],
                    ]
                )
            writer.writerow(
                [
                    self.participant_id,
                    self.seed,
                    self.env.episode,
                    day,
                    self.auto_policy,
                    round(decision_seconds, 3),
                    *state.tolist(),
                    *human_values,
                    *model_action.tolist(),
                ]
            )
        self.rows_saved += 1


class CollectorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("甲公司人类专家数据采集")
        self.geometry("1220x860")
        self.collector = None
        self.participant_var = tk.StringVar(value="anonymous")
        self.output_var = tk.StringVar(value=DEFAULT_OUTPUT)
        self.meta_output_var = tk.StringVar(value=DEFAULT_META_OUTPUT)
        self.status_var = tk.StringVar(value="请阅读任务说明，然后点击“开始采集”。")
        self.action_vars = [tk.StringVar(value="") for _ in ACTION_NAMES]
        self.action_entries = []
        self.adjust_popup = None
        self.suppress_adjust_popup = False
        self.action_hint_vars = [tk.StringVar(value="-") for _ in ACTION_NAMES]
        self.viewing_previous = False
        self.draft_action_values = None
        self.session_started_at = None
        self.current_decision_started_at = None
        self.decision_durations = []
        self.busy = False
        self._build_widgets()
        self._set_action_entries_state(tk.DISABLED)

    def _build_widgets(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill=tk.X)

        ttk.Label(top, text="参与者编号（可选）").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.participant_var, width=18).pack(side=tk.LEFT, padx=(6, 16))
        ttk.Label(top, text="专家数据").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.output_var, width=50).pack(side=tk.LEFT, padx=(6, 16))
        self.start_btn = ttk.Button(top, text="开始采集", command=self.start_collection)
        self.start_btn.pack(side=tk.LEFT)

        path_frame = ttk.Frame(self, padding=(10, 0, 10, 6))
        path_frame.pack(fill=tk.X)
        ttk.Label(path_frame, text="带表头记录").pack(side=tk.LEFT)
        ttk.Entry(path_frame, textvariable=self.meta_output_var, width=92).pack(side=tk.LEFT, padx=(6, 16))

        self.compact_help = ttk.Label(
            self,
            text="任务目标：根据甲公司今天可见的信息，填写贷款、采购和价格，让企业尽量存活更久并保持经营稳定。",
            foreground="#555",
            wraplength=1160,
        )

        self.intro_frame = ttk.Frame(self, padding=30)
        self.intro_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        ttk.Label(
            self.intro_frame,
            text="任务目标",
            font=("Microsoft YaHei", 20, "bold"),
        ).pack(pady=(60, 16))
        ttk.Label(
            self.intro_frame,
            text=(
                "你将扮演“甲公司”的经营决策者。\n\n"
                "每天系统会展示企业现金、库存、债务、还款压力、采购需求和市场报价等关键信息。"
                "请填写今天希望申请的贷款金额、A/B采购需求和销售价格。"
                "系统会自动把你的直观输入转换成模型需要的动作格式，并记录为专家数据。\n\n"
                "目标不是追求某一天的最大收益，而是尽量让企业活得更久、经营更稳定。"
            ),
            font=("Microsoft YaHei", 12),
            justify=tk.CENTER,
            wraplength=760,
        ).pack(pady=10)

        self.main_frame = ttk.Frame(self)

        self.state_table = ttk.Treeview(
            self.main_frame,
            columns=("group", "name", "value", "change"),
            show="headings",
            height=17,
        )
        self.state_table.heading("group", text="类别", anchor=tk.CENTER)
        self.state_table.heading("name", text="信息", anchor=tk.CENTER)
        self.state_table.heading("value", text="参考数值", anchor=tk.CENTER)
        self.state_table.heading("change", text="相对上一天", anchor=tk.CENTER)
        self.state_table.column("group", width=150, anchor=tk.CENTER)
        self.state_table.column("name", width=520, anchor=tk.CENTER)
        self.state_table.column("value", width=180, anchor=tk.CENTER)
        self.state_table.column("change", width=180, anchor=tk.CENTER)
        self.state_table.tag_configure("risk_high", background="#e1f5e6")
        self.state_table.tag_configure("risk_medium", background="#e1f5e6")
        self.state_table.tag_configure("change_good", background="#ffe6e6")
        self.state_table.tag_configure("change_bad", background="#e1f5e6")
        self.state_table.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        action_frame = ttk.LabelFrame(self.main_frame, text="填写今天的决策", padding=10)
        action_frame.pack(fill=tk.X, padx=10)
        descriptions = [
            "0=不申请贷款；最大值=当前现金",
            "直接填写希望采购的原料A数量",
            "直接填写希望采购的原料B数量",
            "直接填写希望设置的产品A销售价格",
        ]
        for idx, name in enumerate(ACTION_DISPLAY_NAMES):
            ttk.Label(action_frame, text=name).grid(row=0, column=idx, sticky=tk.W)
            entry = ttk.Entry(action_frame, textvariable=self.action_vars[idx], width=16, justify=tk.CENTER)
            entry.grid(row=1, column=idx, padx=(0, 18), pady=(2, 4), sticky=tk.W)
            entry.bind("<FocusIn>", lambda event, i=idx: self._show_adjust_popup(i))
            entry.bind("<FocusOut>", lambda event: self.after(120, self._hide_adjust_popup_if_focus_left))
            self.action_vars[idx].trace_add("write", lambda *_: self._update_action_hints())
            self.action_entries.append(entry)
            ttk.Label(action_frame, text=descriptions[idx], foreground="#555", wraplength=250).grid(
                row=2,
                column=idx,
                padx=(0, 18),
                sticky=tk.W,
            )
            ttk.Label(action_frame, textvariable=self.action_hint_vars[idx], foreground="#005a8d", wraplength=250).grid(
                row=3,
                column=idx,
                padx=(0, 18),
                sticky=tk.W,
            )

        button_frame = ttk.Frame(self.main_frame, padding=10)
        button_frame.pack(fill=tk.X)
        self.prev_day_btn = ttk.Button(button_frame, text="查看上一天", command=self.toggle_previous_day, state=tk.DISABLED)
        self.prev_day_btn.pack(side=tk.LEFT, padx=(0, 10))
        self.submit_btn = ttk.Button(
            button_frame,
            text="提交动作并进入下一天",
            command=self.submit_action,
            state=tk.DISABLED,
        )
        self.submit_btn.pack(side=tk.LEFT)
        self.next_episode_btn = ttk.Button(
            button_frame,
            text="开始下一回合",
            command=self.next_episode,
            state=tk.DISABLED,
        )
        self.next_episode_btn.pack(side=tk.LEFT, padx=10)
        ttk.Label(button_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=10)

    def start_collection(self):
        if self.collector is not None:
            self.end_collection()
            return
        self._run_busy("正在初始化环境...", self._start_collection_impl)

    def _start_collection_impl(self):
        self.collector = HumanProductionCollector(
            seed=DEFAULT_SEED,
            output_path=self.output_var.get(),
            meta_output_path=self.meta_output_var.get(),
            auto_policy=DEFAULT_AUTO_POLICY,
            participant_id=self.participant_var.get(),
        )
        self.collector.start_episode()
        self.session_started_at = time.perf_counter()
        self.current_decision_started_at = time.perf_counter()
        self.decision_durations = []
        self.intro_frame.pack_forget()
        self.compact_help.pack(fill=tk.X, padx=12, pady=(0, 8))
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.start_btn.config(text="结束采集")
        self.submit_btn.config(state=tk.NORMAL)
        self.prev_day_btn.config(state=tk.DISABLED)
        self.next_episode_btn.config(state=tk.DISABLED)
        self._set_action_entries_state(tk.NORMAL)
        self._prefill_action_inputs()
        self._refresh_state()
        self._set_status("环境已初始化，请填写今天的决策。")

    def end_collection(self):
        self.submit_btn.config(state=tk.DISABLED)
        self.prev_day_btn.config(state=tk.DISABLED)
        self.next_episode_btn.config(state=tk.DISABLED)
        self._set_action_entries_state(tk.DISABLED)
        self.start_btn.config(state=tk.DISABLED, text="采集已结束")
        self._set_status(f"采集已结束；共保存 {self.collector.rows_saved if self.collector else 0} 条专家数据。")

    def next_episode(self):
        self._run_busy("正在开始下一回合...", self._next_episode_impl)

    def _next_episode_impl(self):
        self.collector.start_episode()
        self.current_decision_started_at = time.perf_counter()
        self.submit_btn.config(state=tk.NORMAL)
        self.prev_day_btn.config(state=tk.DISABLED)
        self.next_episode_btn.config(state=tk.DISABLED)
        self._set_action_entries_state(tk.NORMAL)
        self.viewing_previous = False
        self._prefill_action_inputs()
        self._refresh_state()
        self._set_status("新回合已开始，请填写今天的决策。")

    def submit_action(self):
        self._run_busy("系统正在进入下一天...", self._submit_action_impl)

    def _submit_action_impl(self):
        human_values = self._read_human_values()
        model_action = self._human_to_model_action(human_values)
        now = time.perf_counter()
        decision_seconds = now - self.current_decision_started_at if self.current_decision_started_at else 0
        self.decision_durations.append(decision_seconds)
        done, _ = self.collector.step(model_action, human_values, decision_seconds)
        if done:
            self.submit_btn.config(state=tk.DISABLED)
            self.prev_day_btn.config(state=tk.NORMAL)
            self.next_episode_btn.config(state=tk.NORMAL)
            self._set_action_entries_state(tk.DISABLED)
            self._set_status(
                f"本回合结束，存活 {self.collector.env.day} 天；已保存 {self.collector.rows_saved} 条专家数据。"
            )
            messagebox.showinfo("回合结束", f"甲公司本回合存活 {self.collector.env.day} 天。")
        else:
            self.current_decision_started_at = time.perf_counter()
            self.prev_day_btn.config(state=tk.NORMAL)
            self._prefill_action_inputs()
            self._refresh_state()
            self._set_status(f"已进入第 {self.collector.env.day} 天；已保存 {self.collector.rows_saved} 条专家数据。")

    def toggle_previous_day(self):
        if self.collector is None or not self.collector.history:
            return
        if not self.viewing_previous:
            self.draft_action_values = [var.get() for var in self.action_vars]
            snapshot = self.collector.history[-1]
            for idx, value in enumerate(snapshot["human_values"]):
                self.action_vars[idx].set(format_number(value))
            self._set_action_entries_state(tk.DISABLED)
            self.submit_btn.config(state=tk.DISABLED)
            self.prev_day_btn.config(text="返回今天")
            self.viewing_previous = True
            self._refresh_state(state=snapshot["state"], day=snapshot["day"], readonly=True)
            self._set_status("正在查看上一天记录：这里只能查看，不能修改。")
        else:
            for idx, value in enumerate(self.draft_action_values or []):
                self.action_vars[idx].set(value)
            self._set_action_entries_state(tk.NORMAL)
            self.submit_btn.config(state=tk.NORMAL)
            self.prev_day_btn.config(text="查看上一天")
            self.viewing_previous = False
            self._refresh_state()
            self._set_status("已返回今天，请继续填写今天的决策。")

    def _refresh_state(self, state=None, day=None, readonly=False):
        for item in self.state_table.get_children():
            self.state_table.delete(item)
        if state is None:
            state = self.collector.current_state()
        if day is None:
            day = self.collector.env.day
        previous_state = None
        if not readonly and self.collector.history:
            previous_state = self.collector.history[-1]["state"]
        rows = display_rows(state, day=day)
        for group, name, value, key in rows:
            tags = row_change_tag(key, value, previous_state) or risk_tag(key, value, state)
            change = self._change_text(key, value, previous_state)
            self.state_table.insert("", tk.END, values=(group, name, format_number(value), change), tags=tags)
        mode = "只读查看" if readonly else "当前决策"
        self.state_table.heading("name", text=f"甲公司第 {day} 天可见的信息（{mode}）", anchor=tk.CENTER)

    def _change_text(self, key, value, previous_state):
        if previous_state is None:
            return "-"
        if not isinstance(value, (int, float, np.integer, np.floating)):
            return "-"
        previous = {row[3]: row[2] for row in display_rows(previous_state)}
        if key not in previous:
            return "-"
        if not isinstance(previous[key], (int, float, np.integer, np.floating)):
            return "-"
        delta = value - previous[key]
        if abs(delta) < 1e-9:
            return "无变化"
        sign = "+" if delta > 0 else ""
        return f"{sign}{format_number(delta)}"

    def _prefill_action_inputs(self):
        state = self.collector.current_state()
        defaults = [
            max(0.0, raw_state_value(state, 7)),
            max(0.0, raw_state_value(state, 11)),
            max(0.0, raw_state_value(state, 12)),
            max(0.01, raw_state_value(state, 6)),
        ]
        for idx, value in enumerate(defaults):
            self.action_vars[idx].set(format_number(value))
        self._update_action_hints()

    def _read_human_values(self):
        try:
            values = [float(var.get()) for var in self.action_vars]
        except ValueError as exc:
            raise ValueError("请在四个动作框中输入数字。") from exc
        if any(value < 0 for value in values):
            raise ValueError("请不要输入负数。")
        return values

    def _human_to_model_action(self, human_values):
        state = self.collector.current_state()
        cash = raw_state_value(state, 0)
        k_base = raw_state_value(state, 11)
        l_base = raw_state_value(state, 12)
        price_base = raw_state_value(state, 6)
        loan, k_need, l_need, price = human_values

        if cash <= 0:
            if loan > 0:
                raise ValueError("当前现金为0，贷款意愿只能填写0。")
            loan_action = -0.5
        else:
            if loan > cash:
                raise ValueError(f"贷款意愿不能超过当前现金 {format_number(cash)}。")
            loan_action = loan / cash - 0.5

        k_action = self._quantity_to_action(k_need, k_base, "A采购需求")
        l_action = self._quantity_to_action(l_need, l_base, "B采购需求")
        if price_base <= 0:
            raise ValueError("当前价格基准异常，不能提交价格动作。")
        price_action = price / price_base - 1
        if not -0.5 <= price_action <= 0.5:
            raise ValueError(
                f"销售价格只能在 {format_number(price_base * 0.5)} 到 {format_number(price_base * 1.5)} 之间。"
            )
        return [loan_action, k_action, l_action, price_action]

    def _quantity_to_action(self, value, base, name):
        if base <= 0:
            action = value / 10 - 0.5
            if not -0.5 <= action <= 0.5:
                raise ValueError(f"{name}当前为0，请填写0到10之间的数。")
            return action
        action = value / base - 1
        if not -0.5 <= action <= 0.5:
            raise ValueError(
                f"{name}只能在 {format_number(base * 0.5)} 到 {format_number(base * 1.5)} 之间。"
            )
        return action

    def _update_action_hints(self):
        if self.collector is None:
            for var in self.action_hint_vars:
                var.set("-")
            return
        state = self.collector.current_state()
        cash = raw_state_value(state, 0)
        k_base = raw_state_value(state, 11)
        l_base = raw_state_value(state, 12)
        price_base = raw_state_value(state, 6)
        hints = [
            f"当前现金：{format_number(cash)}；申请贷款金额可填0到{format_number(cash)}",
            f"原料A要和原料B配套；上一回合A参考量 {format_number(k_base)}、B参考量 {format_number(l_base)}。A明显多于B时，多出的A可能无法变成产品。",
            f"原料B要和原料A配套；上一回合B参考量 {format_number(l_base)}、A参考量 {format_number(k_base)}。B明显多于A时，多出的B可能无法变成产品。",
            "产品A定价应兼顾成本收益、乙公司承受能力和双方合作稳定性。",
        ]
        for idx, text in enumerate(hints):
            self.action_hint_vars[idx].set(text)

    def _show_adjust_popup(self, index):
        if self.collector is None or self.viewing_previous:
            return
        if self.suppress_adjust_popup:
            return
        self._destroy_adjust_popup()
        entry = self.action_entries[index]
        x = entry.winfo_rootx() + entry.winfo_width() + 4
        y = entry.winfo_rooty()
        popup = tk.Toplevel(self)
        popup.title("快捷调整")
        popup.geometry(f"+{x}+{y}")
        popup.transient(self)
        popup.resizable(False, False)
        popup.attributes("-topmost", True)
        popup.protocol("WM_DELETE_WINDOW", self._close_adjust_popup_by_user)
        self.adjust_popup = popup

        def apply(kind):
            try:
                current = float(self.action_vars[index].get())
            except ValueError:
                current = 0.0
            if kind == "minus10":
                current *= 0.9
            elif kind == "plus10":
                current *= 1.1
            elif kind == "minus1":
                current -= 1
            elif kind == "plus1":
                current += 1
            elif kind == "zero":
                current = 0
            self.action_vars[index].set(format_number(max(0.0, current)))
            self._destroy_adjust_popup()

        for label, kind in [
            ("减少10%", "minus10"),
            ("增加10%", "plus10"),
            ("-1", "minus1"),
            ("+1", "plus1"),
            ("设为0", "zero"),
        ]:
            ttk.Button(popup, text=label, command=lambda k=kind: apply(k)).pack(side=tk.LEFT, padx=2, pady=4)

        popup.update_idletasks()
        bottom_aligned_y = entry.winfo_rooty() + entry.winfo_height() - popup.winfo_height()
        popup.geometry(f"+{x}+{max(0, bottom_aligned_y)}")

    def _hide_adjust_popup_if_focus_left(self):
        focus = self.focus_get()
        if self.adjust_popup is None:
            return
        if focus in self.action_entries:
            return
        if str(focus).startswith(str(self.adjust_popup)):
            return
        self._destroy_adjust_popup()

    def _close_adjust_popup_by_user(self):
        self.suppress_adjust_popup = True
        self._destroy_adjust_popup()
        self.after(300, self._allow_adjust_popup_again)

    def _allow_adjust_popup_again(self):
        self.suppress_adjust_popup = False

    def _destroy_adjust_popup(self):
        if self.adjust_popup is not None and self.adjust_popup.winfo_exists():
            self.adjust_popup.destroy()
        self.adjust_popup = None

    def _set_action_entries_state(self, state):
        for entry in self.action_entries:
            entry.config(state=state)

    def _run_busy(self, text, func):
        if self.busy:
            return
        self.busy = True
        self._set_status(text)
        self.update_idletasks()
        try:
            func()
        except Exception as exc:
            messagebox.showerror("操作失败", str(exc))
        finally:
            self.busy = False
            self.update_idletasks()

    def _set_status(self, text):
        self.status_var.set(text)


if __name__ == "__main__":
    app = CollectorApp()
    app.mainloop()
