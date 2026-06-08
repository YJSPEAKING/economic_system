def default_actWNDF(enterprise: any, target: str, action: any):  # action: -0.5~0.5
    res = enterprise.__dict__['money'] * (action + 0.5)
    return res


def default_act_shop(enterprise: any, target: str, action: any):  # action: -0.5~0.5
    if enterprise.__dict__['intention_policy'][target] == 0:
        return 10 * (action + 0.5)
    return enterprise.__dict__['intention_policy'][target] * (1 + action)


def default_actPrice(enterprise: any, target: str, action: any):  # action: -0.5~0.5
    res = enterprise.__dict__['next_price']
    enterprise.__dict__['next_price'] = enterprise.__dict__['next_price'] * (1 + action)
    return res


class Enterprise_config:
    def __init__(
        self,
        name: str,
        output_name: str,
        money: float = 0.0,
        WNDF: float = 100.0,
        stock: float = 10.0,
        price: float = 10.0,
        intention: float = 0.0,
        gamma: float = 0.95,
        reward_survival_weight: float = 0.08,
        reward_survival_scale: float = 100.0,
        reward_liquidity_weight: float = 0.25,
        reward_liquidity_scale: float = 1000.0,
        reward_debt_pressure_weight: float = 0.12,
        reward_debt_pressure_cap: float = 2.0,
        fail_reward: float = -10.0,
        fail_reward_clip: float = 18.0,
        fail_survival_target_day: float = 90.0,
        fail_survival_shortfall_weight: float = 8.0,
        action_function: dict = None,
    ):
        self.name = name
        self.output_name = output_name
        self.money = money
        self.WNDF = WNDF
        self.stock = stock
        self.price = price
        self.intention = intention
        self.gamma = gamma
        self.reward_survival_weight = reward_survival_weight
        self.reward_survival_scale = reward_survival_scale
        self.reward_liquidity_weight = reward_liquidity_weight
        self.reward_liquidity_scale = reward_liquidity_scale
        self.reward_debt_pressure_weight = reward_debt_pressure_weight
        self.reward_debt_pressure_cap = reward_debt_pressure_cap
        self.fail_reward = fail_reward
        self.fail_reward_clip = fail_reward_clip
        self.fail_survival_target_day = fail_survival_target_day
        self.fail_survival_shortfall_weight = fail_survival_shortfall_weight
        self.action_function = action_function
        if self.action_function is None:
            self.action_function = {
                'WNDF': default_actWNDF,
                'K': default_act_shop,
                'L': default_act_shop,
                'price': default_actPrice,
            }
