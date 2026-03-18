# 作者：杨逸嘉
# 2026年03月11日17时04分42秒
# 内容：
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
import os


# ==========================================
# 1. 搬运真实环境的网络结构和标准化模块
# ==========================================
class RealActor(nn.Module):
    def __init__(self, s_dim=33, a_dim=4, a_bound=0.5):
        super(RealActor, self).__init__()
        self.l1 = nn.Linear(s_dim, 128)
        self.l2 = nn.Linear(128, 32)
        self.l3 = nn.Linear(32, a_dim)
        self.a_bound = a_bound

    def forward(self, state):
        a = torch.tanh(self.l1(state))
        a = F.leaky_relu(self.l2(a))
        return self.a_bound * torch.tanh(self.l3(a))


class RunningMeanStd:
    def __init__(self, shape=()):
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape)
        self.count = 1e-4

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count


def normalize(x, rms, clamp=True):
    norm_x = (x - rms.mean) / torch.sqrt(rms.var + 1e-8)
    if clamp:
        return torch.clamp(norm_x, -5.0, 5.0)
    return norm_x


# ==========================================
# 2. 开盲盒测试逻辑
# ==========================================
def check_random_actions():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, 'expert_data_production1_cleaned.csv')
    actor_path = os.path.join(current_dir, 'pretrained_actor.pth')

    if not os.path.exists(actor_path):
        print("❌ 找不到 pretrained_actor.pth，请确保预训练脚本已成功运行并保存。")
        return

    print("⏳ 正在加载数据与模型...")
    df = pd.read_csv(csv_path, header=None)
    expert_data = torch.FloatTensor(df.values)

    total_len = len(expert_data)
    train_size = int(total_len * 0.8)

    # 我们只需要恢复训练时的均值和方差统计
    train_states = expert_data[:train_size, :33]
    test_states = expert_data[train_size:, :33]
    test_actions = expert_data[train_size:, 33:37]

    obs_rms = RunningMeanStd(shape=(33,))
    obs_rms.update(train_states)

    # 载入优等生大脑 (统一用 CPU 推理即可)
    actor = RealActor(s_dim=33, a_dim=4, a_bound=0.5)
    actor.load_state_dict(torch.load(actor_path, map_location='cpu'))
    actor.eval()

    print("\n=== 🎯 盲测集随机开盲盒 (脱机预训练 Actor) ===")

    # 随机抽 5 张盲测考卷
    indices = random.sample(range(len(test_states)), 5)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            raw_state = test_states[idx].unsqueeze(0)
            real_a = test_actions[idx].numpy()

            # 必须戴上“标准化的眼镜”才能看懂考题
            norm_state = normalize(raw_state, obs_rms, clamp=True)

            # 模型作答
            fake_a = actor(norm_state).squeeze(0).numpy()

            print(f"\n📦 抽查样本 #{i + 1} (来自数据行 {train_size + idx}):")
            # 保留 4 位小数方便对比
            print(f"✅ 专家真实决策: {np.round(real_a, 4)}")
            print(f"🤖 预训练拟合决策: {np.round(fake_a, 4)}")

            # 计算并输出每个维度的绝对误差
            error = np.abs(real_a - fake_a)
            print(f"📉 各维度绝对误差: {np.round(error, 4)}")


if __name__ == '__main__':
    check_random_actions()