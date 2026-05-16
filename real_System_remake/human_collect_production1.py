import copy
import csv
import os
import random
import sys
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from real_System_remake.Environment import Environment
from real_System_remake.Bank_config import Bank_config
from real_System_remake.Enterprise_config import Enterprise_config


TARGET_AGENT = "production1"
ACTION_NAMES = ["WNDF", "K", "L", "price"]
POLICY_LABEL_TO_VALUE = {
    "TD3自动策略": "td3",
    "固定策略": "fixed",
}
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
    ("上一天K采购需求", 10),
    ("上一天L采购需求", 10),
    ("上一天生产企业1向生产企业1购买价格", 10),
    ("上一天生产企业1向生产企业1购买数量", 10),
    ("上一天生产企业1向消费企业1购买价格", 10),
    ("上一天生产企业1向消费企业1购买数量", 10),
    ("上一天生产企业1向生产第三方市场购买价格", 10),
    ("上一天生产企业1向生产第三方市场购买数量", 10),
    ("上一天生产企业1向消费第三方市场购买价格", 10),
    ("上一天生产企业1向消费第三方市场购买数量", 10),
    ("上一天消费企业1向生产企业1购买价格", 10),
    ("上一天消费企业1向生产企业1购买数量", 10),
    ("上一天消费企业1向消费企业1购买价格", 10),
    ("上一天消费企业1向消费企业1购买数量", 10),
    ("上一天消费企业1向生产第三方市场购买价格", 10),
    ("上一天消费企业1向生产第三方市场购买数量", 10),
    ("上一天消费企业1向消费第三方市场购买价格", 10),
    ("上一天消费企业1向消费第三方市场购买数量", 10),
    ("今天K商品报价：生产企业1", 10),
    ("今天K商品报价：生产第三方市场", 10),
    ("今天L商品报价：消费企业1", 10),
    ("今天L商品报价：消费第三方市场", 10),
]


def state_label(index):
    if index < len(STATE_FIELDS):
        return STATE_FIELDS[index][0]
    return f"未命名状态 {index + 1}"


def display_state_value(index, value):
    scale = STATE_FIELDS[index][1] if index < len(STATE_FIELDS) else 1
    return value * scale


def format_number(value):
    return f"{value:.6g}"


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


class HumanProductionCollector:
    def __init__(
        self,
        seed=184,
        output_path=DEFAULT_OUTPUT,
        meta_output_path=DEFAULT_META_OUTPUT,
        auto_policy="td3",
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
            config = make_enterprise_config(key, output_name)
            self.env.add_enterprise_agent(config=config)
        self.env.add_bank(make_bank_config())
        self.env.add_enterprise_thirdmarket(
            name="production_thirdMarket",
            output_name="K",
            price=100,
        )
        self.env.add_enterprise_thirdmarket(
            name="consumption_thirdMarket",
            output_name="L",
            price=100,
        )
        self.env.init()

    def start_episode(self):
        self.state = self.env.reset()
        self.new_ep = True
        self.auto_agents = {}
        self.history = []
        return self.current_state()

    def current_state(self):
        return np.array(self.state[TARGET_AGENT], dtype=float)

    def _get_auto_agent(self, key):
        if self.auto_policy == "fixed":
            return None

        if key in self.auto_agents:
            return self.auto_agents[key]

        if key in self.env.get_enterprise_execute():
            from real_System_remake.ddpg_enterprise import enterprise_nnu

            config = make_ddpg_config(
                scope=key,
                action_dim=4,
                action_bound=0.5,
                state_dim=len(self.state[key]),
                seed=self.seed,
            )
            agent = enterprise_nnu(config)
        else:
            from real_System_remake.ddpg_bank import bank_nnu

            config = make_ddpg_config(
                scope=key,
                action_dim=2,
                action_bound=0.5,
                state_dim=len(self.state[key]),
                seed=self.seed,
            )
            agent = bank_nnu(config)

        self.auto_agents[key] = agent
        return agent

    def step(self, human_action):
        human_action = np.array(human_action, dtype=float)
        if human_action.shape[0] != len(ACTION_NAMES):
            raise ValueError("production1 action 必须是4个数。")

        action = {TARGET_AGENT: human_action}
        state_before_action = self.current_state().copy()
        day_before_action = self.env.day

        for key in self.env.get_enterprise_execute():
            if key == TARGET_AGENT:
                continue
            if self.auto_policy == "fixed":
                action[key] = FIXED_ENTERPRISE_ACTION.copy()
            else:
                action[key] = self._get_auto_agent(key).run_enterprise(
                    self.state[key],
                    self.new_ep,
                )

        for key in self.env.get_bank_execute():
            if self.auto_policy == "fixed":
                action[key] = FIXED_BANK_ACTION.copy()
            else:
                action[key] = self._get_auto_agent(key).run_bank(
                    self.state[key],
                    self.new_ep,
                )

        self.history.append(
            {
                "day": day_before_action,
                "state": state_before_action,
                "action": human_action.copy(),
            }
        )
        self._save_row(state_before_action, human_action, day_before_action)
        self.env.step(action)
        next_state, reward, done = self.env.observe()

        self.state = next_state
        self.new_ep = False
        return done, reward

    def _save_row(self, state, action, day):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([*state.tolist(), *action.tolist()])

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
                        *[f"state_{i}_{state_label(i)}" for i in range(len(state))],
                        *[f"action_{name}" for name in ACTION_NAMES],
                    ]
                )
            writer.writerow(
                [
                    self.participant_id,
                    self.seed,
                    self.env.episode,
                    day,
                    self.auto_policy,
                    *state.tolist(),
                    *action.tolist(),
                ]
            )
        self.rows_saved += 1


class CollectorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("生产企业1 人类专家数据采集")
        self.geometry("1220x840")
        self.collector = None
        self.action_vars = [tk.StringVar(value="0.0") for _ in ACTION_NAMES]
        self.action_entries = []
        self.preview_vars = [tk.StringVar(value="-") for _ in ACTION_NAMES]
        self.participant_var = tk.StringVar(value="anonymous")
        self.seed_var = tk.StringVar(value="184")
        self.auto_policy_var = tk.StringVar(value="TD3自动策略")
        self.output_var = tk.StringVar(value=DEFAULT_OUTPUT)
        self.meta_output_var = tk.StringVar(value=DEFAULT_META_OUTPUT)
        self.status_var = tk.StringVar(value="点击“开始采集”初始化环境。")
        self.viewing_previous = False
        self.draft_action_values = None

        self._build_widgets()

    def _build_widgets(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill=tk.X)

        ttk.Label(top, text="参与者").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.participant_var, width=14).pack(side=tk.LEFT, padx=(6, 16))
        ttk.Label(top, text="Seed").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.seed_var, width=8).pack(side=tk.LEFT, padx=(6, 16))
        ttk.Label(top, text="非人工主体策略").pack(side=tk.LEFT)
        ttk.Combobox(
            top,
            textvariable=self.auto_policy_var,
            values=tuple(POLICY_LABEL_TO_VALUE.keys()),
            state="readonly",
            width=12,
        ).pack(side=tk.LEFT, padx=(6, 16))
        ttk.Label(top, text="专家数据").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.output_var, width=48).pack(side=tk.LEFT, padx=(6, 16))
        ttk.Button(top, text="开始采集", command=self.start_collection).pack(side=tk.LEFT)

        path_frame = ttk.Frame(self, padding=(10, 0, 10, 6))
        path_frame.pack(fill=tk.X)
        ttk.Label(path_frame, text="带表头记录").pack(side=tk.LEFT)
        ttk.Entry(path_frame, textvariable=self.meta_output_var, width=88).pack(side=tk.LEFT, padx=(6, 16))

        help_text = (
            "任务目标：请根据生产企业1今天可见的信息，选择今天的贷款、采购和价格动作，尽量让企业存活更久并保持经营稳定。\n"
            "非人工主体策略：TD3自动策略更接近当前实验系统，会让消费企业1和银行自动决策；"
            "固定策略只用于缺少模型依赖时兜底，可能明显改变存活天数。"
        )
        ttk.Label(self, text=help_text, foreground="#555", wraplength=1160).pack(
            fill=tk.X,
            padx=12,
            pady=(0, 8),
        )

        self.state_table = ttk.Treeview(
            self,
            columns=("idx", "name", "value"),
            show="headings",
            height=24,
        )
        self.state_table.heading("idx", text="#", anchor=tk.CENTER)
        self.state_table.heading("name", text="生产企业1今天可见的信息", anchor=tk.CENTER)
        self.state_table.heading("value", text="数值", anchor=tk.CENTER)
        self.state_table.column("idx", width=60, anchor=tk.CENTER)
        self.state_table.column("name", width=520, anchor=tk.CENTER)
        self.state_table.column("value", width=180, anchor=tk.CENTER)
        self.state_table.tag_configure("risk_high", background="#ffe0e0")
        self.state_table.tag_configure("risk_medium", background="#fff2c2")
        self.state_table.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        action_frame = ttk.LabelFrame(self, text="输入生产企业1今日动作，建议范围 -0.5 到 0.5", padding=10)
        action_frame.pack(fill=tk.X, padx=10)
        action_descriptions = [
            "贷款意愿：-0.5=不申请，0=约现金50%，0.5=约现金100%",
            "K采购需求：0=保持，0.1=增加10%，-0.1=减少10%",
            "L采购需求：0=保持，0.1=增加10%，-0.1=减少10%",
            "价格调整：0=明天价格不变，0.1=明天涨价10%，-0.1=降价10%",
        ]
        for idx, name in enumerate(ACTION_NAMES):
            ttk.Label(action_frame, text=name).grid(row=0, column=idx, sticky=tk.W)
            entry = ttk.Entry(action_frame, textvariable=self.action_vars[idx], width=14, justify=tk.CENTER)
            entry.grid(row=1, column=idx, padx=(0, 18), pady=(2, 4), sticky=tk.W)
            self.action_vars[idx].trace_add("write", lambda *_: self._update_action_preview())
            self.action_entries.append(entry)
            ttk.Label(action_frame, text=action_descriptions[idx], foreground="#555", wraplength=250).grid(
                row=2,
                column=idx,
                padx=(0, 18),
                sticky=tk.W,
            )
            ttk.Label(action_frame, textvariable=self.preview_vars[idx], foreground="#005a8d", wraplength=250).grid(
                row=3,
                column=idx,
                padx=(0, 18),
                sticky=tk.W,
            )

        button_frame = ttk.Frame(self, padding=10)
        button_frame.pack(fill=tk.X)
        self.prev_day_btn = ttk.Button(
            button_frame,
            text="查看上一天",
            command=self.toggle_previous_day,
            state=tk.DISABLED,
        )
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
        try:
            self.collector = HumanProductionCollector(
                seed=int(self.seed_var.get()),
                output_path=self.output_var.get(),
                meta_output_path=self.meta_output_var.get(),
                auto_policy=POLICY_LABEL_TO_VALUE[self.auto_policy_var.get()],
                participant_id=self.participant_var.get(),
            )
            self.collector.start_episode()
        except Exception as exc:
            messagebox.showerror("初始化失败", str(exc))
            return

        self.submit_btn.config(state=tk.NORMAL)
        self.prev_day_btn.config(state=tk.DISABLED)
        self.next_episode_btn.config(state=tk.DISABLED)
        self.viewing_previous = False
        self._refresh_state()
        self._update_action_preview()
        self._set_status("环境已初始化，请输入 production1 今日动作。")

    def next_episode(self):
        self.collector.start_episode()
        self.submit_btn.config(state=tk.NORMAL)
        self.prev_day_btn.config(state=tk.DISABLED)
        self.next_episode_btn.config(state=tk.DISABLED)
        self.viewing_previous = False
        self._refresh_state()
        self._update_action_preview()
        self._set_status("新回合已开始。")

    def toggle_previous_day(self):
        if self.collector is None or not self.collector.history:
            return

        if not self.viewing_previous:
            self.draft_action_values = [var.get() for var in self.action_vars]
            snapshot = self.collector.history[-1]
            for idx, value in enumerate(snapshot["action"]):
                self.action_vars[idx].set(f"{value:.6g}")
            self._set_action_entries_state(tk.DISABLED)
            self._update_action_preview(readonly=True)
            self.submit_btn.config(state=tk.DISABLED)
            self.prev_day_btn.config(text="返回今天")
            self.viewing_previous = True
            self._refresh_state(state=snapshot["state"], day=snapshot["day"], readonly=True)
            self._set_status("正在查看上一天记录：这里只能查看，不能修改动作。")
        else:
            if self.draft_action_values is not None:
                for idx, value in enumerate(self.draft_action_values):
                    self.action_vars[idx].set(value)
            self._set_action_entries_state(tk.NORMAL)
            self.submit_btn.config(state=tk.NORMAL)
            self.prev_day_btn.config(text="查看上一天")
            self.viewing_previous = False
            self._refresh_state()
            self._update_action_preview()
            self._set_status("已返回今天，请继续输入生产企业1今日动作。")

    def submit_action(self):
        if self.collector is None:
            return
        try:
            action = [float(var.get()) for var in self.action_vars]
            done, reward = self.collector.step(action)
        except Exception as exc:
            messagebox.showerror("动作提交失败", str(exc))
            return

        if done:
            self.submit_btn.config(state=tk.DISABLED)
            self.prev_day_btn.config(state=tk.NORMAL)
            self.next_episode_btn.config(state=tk.NORMAL)
            self._set_status(
                f"本回合结束，存活 {self.collector.env.day} 天；"
                f"已保存 {self.collector.rows_saved} 条专家数据。"
            )
            messagebox.showinfo("回合结束", f"production1 本回合存活 {self.collector.env.day} 天。")
        else:
            self.prev_day_btn.config(state=tk.NORMAL)
            self._refresh_state()
            self._update_action_preview()
            self._set_status(
                f"已进入第 {self.collector.env.day} 天；"
                f"已保存 {self.collector.rows_saved} 条专家数据。"
            )

    def _refresh_state(self, state=None, day=None, readonly=False):
        for item in self.state_table.get_children():
            self.state_table.delete(item)
        if state is None:
            state = self.collector.current_state()
        if day is None and self.collector is not None:
            day = self.collector.env.day
        for idx, value in enumerate(state):
            shown_value = display_state_value(idx, value)
            tags = self._state_tags(idx, shown_value, state)
            self.state_table.insert(
                "",
                tk.END,
                values=(idx, state_label(idx), format_number(shown_value)),
                tags=tags,
            )
        mode = "只读查看" if readonly else "当前决策"
        self.state_table.heading("name", text=f"生产企业1第 {day} 天可见的信息（{mode}）")

    def _set_action_entries_state(self, state):
        for entry in self.action_entries:
            entry.config(state=state)

    def _state_tags(self, idx, value, state):
        cash = display_state_value(0, state[0])
        if idx == 0:
            if value < 50:
                return ("risk_high",)
            if value < 200:
                return ("risk_medium",)
        if idx == 2:
            if cash > 0 and value > cash * 3:
                return ("risk_high",)
            if cash > 0 and value > cash:
                return ("risk_medium",)
        if idx in (9, 10):
            if cash > 0 and value > cash:
                return ("risk_high",)
            if cash > 0 and value > cash * 0.5:
                return ("risk_medium",)
        return ()

    def _update_action_preview(self, readonly=False):
        if self.collector is None:
            for var in self.preview_vars:
                var.set("-")
            return

        state = self.collector.current_state()
        cash = display_state_value(0, state[0])
        k_need = display_state_value(11, state[11])
        l_need = display_state_value(12, state[12])
        next_price = display_state_value(6, state[6])

        try:
            values = [float(var.get()) for var in self.action_vars]
        except ValueError:
            for var in self.preview_vars:
                var.set("请输入数字")
            return

        wndef, k_action, l_action, price_action = values
        previews = [
            f"预计贷款意愿约 {format_number(cash * (wndef + 0.5))}",
            f"预计K需求约 {format_number((10 if k_need == 0 else k_need) * (k_action + 0.5 if k_need == 0 else 1 + k_action))}",
            f"预计L需求约 {format_number((10 if l_need == 0 else l_need) * (l_action + 0.5 if l_need == 0 else 1 + l_action))}",
            f"预计明天报价约 {format_number(next_price * (1 + price_action))}",
        ]
        if readonly:
            previews = [f"上一天记录：{text}" for text in previews]
        for idx, text in enumerate(previews):
            self.preview_vars[idx].set(text)

    def _set_status(self, text):
        self.status_var.set(text)


if __name__ == "__main__":
    app = CollectorApp()
    app.mainloop()
