def default_actWNDB(bank: any, target: str, action: any):  # action:0~1
    res = bank.__dict__['observation'][target].WNDF * (action)
    # if bank.observation[target].WNDF == 100.0:
    #     res = 100.0
    return res


class Bank_config:
    def __init__(self,
                 name: str,
                 fund: float = 100,
                 fund_rate: float = 1,
                 fund_increase: float = 0.1,
                 debt_time: int = 5,
                 debt_i: float = 0.005,
                 reward_profit_scale: float = 100.0,
                 reward_exposure_scale: float = 1000.0,
                 reward_profit_weight: float = 1.0,
                 reward_fill_weight: float = 0.2,
                 reward_exposure_weight: float = 0.03,
                 reward_alive_weight: float = 0.05,
                 reward_survival_weight: float = 0.15,
                 reward_survival_scale: float = 100.0,
                 reward_survival_power: float = 2.0,
                 reward_survival_milestone_weight: float = 0.15,
                 reward_survival_milestone_day: float = 80.0,
                 reward_liquidity_weight: float = 0.1,
                 reward_liquidity_scale: float = 1000.0,
                 reward_due_shortage_weight: float = 0.35,
                 reward_due_shortage_cap: float = 1.5,
                 reward_clip: float = 5.0,
                 fail_reward: float = -6.0,
                 fail_reward_clip: float = 12.0,
                 fail_survival_target_day: float = 90.0,
                 fail_survival_shortfall_weight: float = 6.0,
                 fail_exposure_weight: float = 0.12,
                 action_function: dict = None,  # 类型为字典 str:function()，决定设置各变量的方法
                 ):
        self.name = name
        self.fund = fund
        self.fund_rate = fund_rate
        self.fund_increase = fund_increase
        self.debt_time = debt_time
        self.debt_i = debt_i
        self.reward_profit_scale = reward_profit_scale
        self.reward_exposure_scale = reward_exposure_scale
        self.reward_profit_weight = reward_profit_weight
        self.reward_fill_weight = reward_fill_weight
        self.reward_exposure_weight = reward_exposure_weight
        self.reward_alive_weight = reward_alive_weight
        self.reward_survival_weight = reward_survival_weight
        self.reward_survival_scale = reward_survival_scale
        self.reward_survival_power = reward_survival_power
        self.reward_survival_milestone_weight = reward_survival_milestone_weight
        self.reward_survival_milestone_day = reward_survival_milestone_day
        self.reward_liquidity_weight = reward_liquidity_weight
        self.reward_liquidity_scale = reward_liquidity_scale
        self.reward_due_shortage_weight = reward_due_shortage_weight
        self.reward_due_shortage_cap = reward_due_shortage_cap
        self.reward_clip = reward_clip
        self.fail_reward = fail_reward
        self.fail_reward_clip = fail_reward_clip
        self.fail_survival_target_day = fail_survival_target_day
        self.fail_survival_shortfall_weight = fail_survival_shortfall_weight
        self.fail_exposure_weight = fail_exposure_weight
        self.action_function = action_function
        if self.action_function is None:
            self.action_function = {
                                    'WNDB':default_actWNDB,
                                    }








