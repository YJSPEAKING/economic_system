from .Enterprise import Enterprise
from .Bank_config import Bank_config
import math
import random
# 银行参数设置在外面，一般来说是不用改的
M = 0  # 现金（等价于利润）
Ω = 1  # Ω 分别为一、二类企业债券 数据格式为列表
D = 2  # D 分别为银行对一、二类企业的欠款 数据格式为列表
WNDB = 3  # 分别为一、二类企业贷款意愿 数据格式为列表
real_WNDB = 4  # 分别为一、二类企业执行侧贷款意愿 数据格式为列表
OB = 5  # 分别为一、二类企业观察，对应数据项应为列表，是企业观察，对应数据为e_data
bank_current = 6


class Bank:
    def __init__(self,
                 config: Bank_config
                 ):                          # 字典数据全部以 （主体名称，数据内容）格式存储
        self.config = config
        self.name = config.name
        self.money = 0                       # M 银行现金，即利润总和
        self.profit = 0                      # 利润，等价于当前回合收回利息数，或破产后减少的金额
        self.total_profit = 0
        self.able_fund = 0
        self.debt = {}                       # D 银行对企业的欠款，即 企业现金
        self.bond = {}                       # Ω 银行对企业的债券，即 企业欠银 行的钱
        self.bond_detail = {}                # 待还款细则 为{(主体名:str,list[debt_time])}
        self.should_payback = {}             # 当日待还款数值 {(主体名:str,待还款:float)}

        self.WNDB = {}                       # 该回合银行决策对企业的放贷
        self.real_WNDB = {}                  # 该回合银行实际执行对企业的放贷
        self.observation = {}                # 银行对企业的观察，为引用传递，设定完观察无需修改
        self.loss = {}
        self.reward = {}

        # 以上为数据，以下为参数
        self.debt_time = config.debt_time
        self.debet_i = config.debt_i
        self.fund = self.config.fund                # 银行储备金
        self.fund_rate = config.fund_rate           # 储备金率，银行总共可以放出的贷款额度为 sum(bond) <= fund_rate * (fund + money)
        self.fund_increase = config.fund_increase   # 储备金每回合增长，即 fund = fund * (1 + fund_increase) ^ day
        self.action_function = config.action_function
        self.reward_profit_weight = getattr(config, 'reward_profit_weight', 1.0)
        self.reward_credit_weight = getattr(config, 'reward_credit_weight', 0.3)
        self.reward_survival_weight = getattr(config, 'reward_survival_weight', 0.05)
        self.reward_unmet_credit_weight = getattr(config, 'reward_unmet_credit_weight', 0.1)
        self.reward_default_weight = getattr(config, 'reward_default_weight', 1.0)
        self.reward_smooth_weight = getattr(config, 'reward_smooth_weight', 0.1)
        self.reward_value_scale = getattr(config, 'reward_value_scale', 100.0)
        self.last_total_real_WNDB = 0.0
        self.total_reward = {'WNDB': 0, }           # 如果要更改奖励就在这里改
        self.reward_decay = 0.95
        self.step = 0

    # 观察企业的负债 资金 还款 贷款意愿 实际贷款意愿 企业对银行借款详情
    def observe(self, target: Enterprise):
        self.observation[target.name] = target
        self.debt[target.name] = 0
        self.bond[target.name] = 0
        self.WNDB[target.name] = 0
        self.real_WNDB[target.name] = 0
        self.bond_detail[target.name] = [0 for x in range(self.debt_time)]
        self.should_payback[target.name] = 0

    # 新的回合 某个企业破产或达到预制天数上线T 一回合结束 重置环境和智能体的属性
    def new_episode(self):
        self.money = 0  # M 银行现金，即利润总和
        self.profit = 0  # 利润，等价于当前回合收回利息数，或破产后减少的金额
        self.total_profit = 0
        self.able_fund = 0
        self.debt = {}  # D 银行对企业的欠款，即 企业现金
        self.bond = {}  # Ω 银行对企业的债券，即 企业欠银 行的钱
        self.bond_detail = {}  # 待还款细则 为{(主体名:str,list[debt_time])}
        self.should_payback = {}  # 当日待还款数值 {(主体名:str,待还款:float)}
        self.loss = {}
        self.reward = {}
        self.total_reward = {'WNDB': 0}
        self.step = 0
        self.last_total_real_WNDB = 0.0

        self.WNDB = {}  # 该回合银行决策对企业的放贷
        self.real_WNDB = {}  # 该回合银行实际执行对企业的放贷
        self.fund = self.config.fund                     # 银行储备金

        # keys：企业1，企业2
        for key in self.observation.keys():
            self.debt[key] = 0
            self.bond[key] = 0
            self.WNDB[key] = 0
            self.real_WNDB[key] = 0
            self.bond_detail[key] = [0 for x in range(self.debt_time)]
            self.should_payback[key] = 0

    def new_day(self):
        for key in self.observation.keys():
            self.WNDB[key] = 0
            self.real_WNDB[key] = 0
            self.should_payback[key] = 0
        self.develop()
        self.profit = 0  # 利润，等价于当前回合收回利息数，或破产后减少的金额

    def set_action(self,
               target:str,
               action:any,
               actionType: str = 'WNDB'):
        try:
            self.WNDB[target] = self.action_function[actionType](self, target, action)
        except KeyError:    # actionType方法不存在 则直接赋值
            self.WNDB[target] = action

    def check_rent(self):
        self.able_fund = round(self.fund_rate * (self.money + self.fund)-sum(self.bond.values()), 2)
        total_WNDB = sum(self.WNDB.values())
        percent = min(self.able_fund/(total_WNDB + 1e-2), 1)   # 储备金不够总待借出，则按比例
        for key in self.real_WNDB.keys():
            self.real_WNDB[key] = round(self.WNDB[key] * percent, 2)


    def rent(self, name: str, day:int) -> float:
        rent_val = self.real_WNDB[name]
        self.bond_detail[name][day % self.debt_time] = rent_val
        self.debt[name] += rent_val          # 银行给出现金，银行对企业债务加一笔
        self.bond[name] += rent_val          # 银行放出贷款，银行对企业债权加一笔
        return rent_val

    def answer_debt(self, name: str, day):   # 回合开始由企业询问该回合待还款本金和待还款利息
        self.should_payback[name] = self.bond_detail[name][day % self.debt_time]  # 记录下当日待还款
        return {'money': round(self.should_payback[name], 2), 'iD':round(self.debet_i * self.bond[name], 2)}

    def deal_payback(self, name:str, payback: dict):   # 处理来自企业的还款
        self.debt[name] = round(self.debt[name] - payback['payback'], 2)      # 银行收回现金，银行对企业债务减一笔
        self.bond[name] = round(self.bond[name] - payback['payback'], 2)      # 银行收回贷款，银行对企业债权减一笔
        self.profit = round(self.profit + payback['iD'], 2)                    # 银行收回利息，银行利润加一笔
        self.total_profit += self.profit
        self.money = round(self.money + payback['iD'], 2)                     # 银行收回利息，银行现金（总利润）加一笔
        self.debt[name] = round(self.debt[name] - payback['iD'], 2)           # 银行收回利息，银行对企业欠款减少

    def trade_callback(self, name: str, delta_money:float):
        # 企业处理完订单后变动的金钱数量，平衡资产债务表
        self.debt[name] = round(self.debt[name] + delta_money, 2)



    def develop(self):  # 经济发展，储备金增加
        # 根据市面上流动的现金来决定，市面上现金为银行对企业债务之和
        self.fund += sum(self.debt.values()) * self.fund_increase

    def custom_state(self):
        state = [self.money/100, self.able_fund/1000]
        flag = True
        for key in self.observation.keys():
            if flag:
                ns = self.observation[key].get_state()
                flag = False
            else:
                ns = self.observation[key].get_state()[0:13]
            state = state + ns
        # for key in self.observation.keys():
        #     state = state + self.observation[key].get_state()
        self.state = state

    def get_state(self):
        return self.state

    def _scale_reward_value(self, value):
        return math.tanh(value / (self.reward_value_scale + 1e-6))


    def custom_reward(self):
        total_wndb = sum(max(value, 0) for value in self.WNDB.values())
        total_real_wndb = sum(max(value, 0) for value in self.real_WNDB.values())
        credit_fill_rate = total_real_wndb / (total_wndb + 1e-6) if total_wndb > 1e-6 else 0
        unmet_rate = max(0, 1 - credit_fill_rate) if total_wndb > 1e-6 else 0
        credit_volatility = abs(total_real_wndb - self.last_total_real_WNDB)

        interest_reward = self.reward_profit_weight * self._scale_reward_value(self.profit)
        credit_support = self.reward_credit_weight * credit_fill_rate
        unmet_penalty = self.reward_unmet_credit_weight * unmet_rate
        smooth_penalty = self.reward_smooth_weight * self._scale_reward_value(credit_volatility)
        alive_count = sum(0 if target.is_falled() else 1 for target in self.observation.values())
        survival_reward = self.reward_survival_weight * alive_count
        default_exposure = sum(self.bond[key] for key, target in self.observation.items() if target.is_falled())
        default_penalty = self.reward_default_weight * self._scale_reward_value(default_exposure)
        self.reward['WNDB'] = interest_reward + credit_support + survival_reward - unmet_penalty - smooth_penalty - default_penalty
        self.last_total_real_WNDB = total_real_wndb


    def get_reward(self):
        decay = self.reward_decay ** self.step
        for key in self.total_reward:
            self.total_reward[key] += self.reward[key] * decay
        self.step += 1
        return self.reward

    def get_fail_reward(self):
        fail_reward = {'WNDB':0}
        decay = self.reward_decay ** self.step
        default_exposure = sum(self.bond[key] for key, target in self.observation.items() if target.is_falled())
        fail_reward['WNDB'] = -10 - self.reward_default_weight * self._scale_reward_value(default_exposure)
        self.total_reward['WNDB'] += fail_reward['WNDB'] * decay
        return fail_reward

    def final_settlement(self):
        for key in self.observation:
            # 如果某企业破产
            # 因为还款时现金已经清零了
            # 此时理论上银行对企业债务debt为0
            # 此时剩余的银行对企业的债权bond，就是企业欠银行且还不上的贷款
            # 这部分需要银行自掏腰包
            if self.observation[key].is_falled():
                delta_money = self.bond[key]
                self.profit -= delta_money
                self.total_profit += self.profit
                self.money -= delta_money



    def __str__(self):
        res = self.name + \
                    "\n 银行利润：" + str(self.money) +\
                    " 银行债券：" + str(self.bond) +\
                    " 银行欠债：" + str(self.debt) +\
                    " 银行贷款意愿：" + str(self.WNDB) + \
                    " 银行实际贷款意愿：" + str(self.real_WNDB) + \
                    "\n 储备金:" + str(self.fund_rate * (self.money + self.fund)) +\
                    " 剩余可用储备金:" + str(round(self.fund_rate * (self.money + self.fund)-sum(self.bond.values()), 2)) + \
                    " 借款详情:" + str(self.bond_detail) +'\n'
        for key in self.real_WNDB:
            res = res + '银行实际借给' + str(key) + " " + str(self.real_WNDB[key]) + " 的贷款\n"
        return res
