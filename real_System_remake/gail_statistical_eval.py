import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 解决库冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# ==========================================
# 0. 定义与预训练脚本完全一致的模型结构
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


class RealDiscriminator(nn.Module):
    def __init__(self, s_dim=33, a_dim=4, hidden_size=100):
        super(RealDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + a_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))


# ==========================================
# 1. 核心统计工具 (JS 散度 V2 版)
# ==========================================
def calculate_metrics(dist1, dist2):
    w_dist = wasserstein_distance(dist1, dist2)
    # 使用直方图频率分布计算 JS 散度
    p, _ = np.histogram(dist1, bins=100, range=(0, 1))
    q, _ = np.histogram(dist2, bins=100, range=(0, 1))
    p = p / (np.sum(p) + 1e-10)
    q = q / (np.sum(q) + 1e-10)
    js_div = distance.jensenshannon(p, q) ** 2
    return w_dist, js_div


def run_standalone_analysis():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. 加载已保存的参数和模型
    print("⏳ 正在加载预训练组件...")
    try:
        rms_params = torch.load(os.path.join(current_dir, 'obs_rms_params.pth'), map_location=device)
        obs_mean, obs_var = rms_params['mean'].to(device), rms_params['var'].to(device)
        # 核心：必须加载动作标准化参数才能复现结果
        act_mean, act_var = rms_params['act_mean'].to(device), rms_params['act_var'].to(device)

        actor = RealActor().to(device)
        actor.load_state_dict(torch.load(os.path.join(current_dir, 'pretrained_actor.pth'), map_location=device))

        disc = RealDiscriminator().to(device)
        disc.load_state_dict(torch.load(os.path.join(current_dir, 'pretrained_discriminator.pth'), map_location=device))

        actor.eval();
        disc.eval()
        print("✅ 组件加载完毕。")
    except KeyError:
        print("❌ 错误：obs_rms_params.pth 缺少 act_mean/act_var。请确保预训练脚本保存了全部参数。")
        return

    # 3. 加载测试数据 (CSV 后 20%)
    df = pd.read_csv(os.path.join(current_dir, 'expert_data_production1_cleaned.csv'), header=None)
    test_raw = torch.FloatTensor(df.values[int(len(df) * 0.8):]).to(device)
    test_s, test_a = test_raw[:, :33], test_raw[:, 33:37]

    # 4. 【完全复刻】数据处理逻辑
    with torch.no_grad():
        # 状态标准化
        s_norm = torch.clamp((test_s - obs_mean) / torch.sqrt(obs_var + 1e-8), -5.0, 5.0)

        # 专家得分 (Action 标准化)
        a_real_norm = (test_a - act_mean) / torch.sqrt(act_var + 1e-8)
        real_scores = torch.sigmoid(disc(s_norm, a_real_norm)).cpu().numpy().flatten()

        # Agent 得分
        fake_a = actor(s_norm)
        a_fake_norm = (fake_a - act_mean) / torch.sqrt(act_var + 1e-8)
        fake_scores = torch.sigmoid(disc(s_norm, a_fake_norm)).cpu().numpy().flatten()

        # Noise 得分
        noise_a = (torch.rand_like(test_a) * 2 - 1) * 0.5
        a_noise_norm = (noise_a - act_mean) / torch.sqrt(act_var + 1e-8)
        noise_scores = torch.sigmoid(disc(s_norm, a_noise_norm)).cpu().numpy().flatten()

    # 5. 统计分析
    w_fake, js_fake = calculate_metrics(real_scores, fake_scores)
    w_noise, js_noise = calculate_metrics(real_scores, noise_scores)

    # 6. 绘图
    plt.figure(figsize=(10, 6))
    plt.hist(real_scores, bins=50, alpha=0.6, label='Expert (Real)', color='green', range=(0, 1))
    plt.hist(fake_scores, bins=50, alpha=0.6, label=f'Agent (JS:{js_fake:.4f})', color='red', range=(0, 1))
    plt.hist(noise_scores, bins=50, alpha=0.3, label=f'Noise (JS:{js_noise:.4f})', color='gray', range=(0, 1))
    plt.title('Post-Pretrain Data Analysis')
    plt.xlabel('Discriminator Score');
    plt.ylabel('Count');
    plt.legend();
    plt.grid(True, alpha=0.2)

    plt.savefig(os.path.join(current_dir, 'analysis_report.png'), dpi=300)
    print(f"\n📊 分析完成！报告已保存。")
    print(f"🔹 JS Divergence: Agent({js_fake:.4f}) | Noise({js_noise:.4f})")
    print(f"🔹 Wasserstein:    Agent({w_fake:.4f}) | Noise({w_noise:.4f})")


if __name__ == '__main__':
    run_standalone_analysis()