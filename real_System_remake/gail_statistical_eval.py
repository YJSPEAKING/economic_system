import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from scipy.stats import wasserstein_distance, ks_2samp
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
# 1. 核心统计工具 (加入重采样 P 值计算)
# ==========================================
def calculate_metrics_refined(dist_ref, dist_target, resample_size=500, n_iterations=20):
    # 1. Wasserstein 距离
    w_dist = wasserstein_distance(dist_ref, dist_target)

    # 2. JS 散度
    p_hist, _ = np.histogram(dist_ref, bins=100, range=(0, 1))
    q_hist, _ = np.histogram(dist_target, bins=100, range=(0, 1))
    p_prob = p_hist / (np.sum(p_hist) + 1e-10)
    q_prob = q_hist / (np.sum(q_hist) + 1e-10)
    js_div = distance.jensenshannon(p_prob, q_prob) ** 2

    # 3. 重采样 KS 检验 (解决样本量过大导致 P 值失效的问题)
    p_values = []
    ks_stats = []

    # 确保采样量不超过实际数据量
    sample_n = min(resample_size, len(dist_ref), len(dist_target))

    for _ in range(n_iterations):
        idx_ref = np.random.choice(len(dist_ref), sample_n, replace=False)
        idx_target = np.random.choice(len(dist_target), sample_n, replace=False)

        stat, p_val = ks_2samp(dist_ref[idx_ref], dist_target[idx_target])
        ks_stats.append(stat)
        p_values.append(p_val)

    # 取中位数作为稳健的评估指标
    final_p = np.median(p_values)
    final_ks = np.median(ks_stats)

    return w_dist, js_div, final_p, final_ks


def run_standalone_analysis():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_dir = os.path.dirname(os.path.abspath(__file__))

    print("⏳ 正在加载预训练组件...")
    try:
        rms_params = torch.load(os.path.join(current_dir, 'obs_rms_params.pth'), map_location=device)
        obs_mean, obs_var = rms_params['mean'].to(device), rms_params['var'].to(device)
        act_mean, act_var = rms_params['act_mean'].to(device), rms_params['act_var'].to(device)

        actor = RealActor().to(device)
        actor.load_state_dict(torch.load(os.path.join(current_dir, 'pretrained_actor.pth'), map_location=device))

        disc = RealDiscriminator().to(device)
        disc.load_state_dict(torch.load(os.path.join(current_dir, 'pretrained_discriminator.pth'), map_location=device))

        actor.eval();
        disc.eval()
        print("✅ 组件加载完毕。")
    except Exception as e:
        print(f"❌ 错误：{e}")
        return

    df = pd.read_csv(os.path.join(current_dir, 'expert_data_production1_cleaned.csv'), header=None)
    test_raw = torch.FloatTensor(df.values[int(len(df) * 0.8):]).to(device)
    test_s, test_a = test_raw[:, :33], test_raw[:, 33:37]

    with torch.no_grad():
        s_norm = torch.clamp((test_s - obs_mean) / torch.sqrt(obs_var + 1e-8), -5.0, 5.0)
        a_real_norm = (test_a - act_mean) / torch.sqrt(act_var + 1e-8)
        real_scores = torch.sigmoid(disc(s_norm, a_real_norm)).cpu().numpy().flatten()

        fake_a = actor(s_norm)
        a_fake_norm = (fake_a - act_mean) / torch.sqrt(act_var + 1e-8)
        fake_scores = torch.sigmoid(disc(s_norm, a_fake_norm)).cpu().numpy().flatten()

        noise_a = (torch.rand_like(test_a) * 2 - 1) * 0.5
        a_noise_norm = (noise_a - act_mean) / torch.sqrt(act_var + 1e-8)
        noise_scores = torch.sigmoid(disc(s_norm, a_noise_norm)).cpu().numpy().flatten()

    # 执行精细化统计分析
    w_fake, js_fake, p_fake, ks_fake = calculate_metrics_refined(real_scores, fake_scores)
    w_noise, js_noise, p_noise, ks_noise = calculate_metrics_refined(real_scores, noise_scores)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.hist(real_scores, bins=50, alpha=0.6, label='Expert (Real)', color='green', range=(0, 1))
    plt.hist(fake_scores, bins=50, alpha=0.6, label=f'Agent (JS:{js_fake:.3f}, P:{p_fake:.2e})', color='red',
             range=(0, 1))
    plt.hist(noise_scores, bins=50, alpha=0.3, label=f'Noise (JS:{js_noise:.3f}, P:{p_noise:.2e})', color='gray',
             range=(0, 1))
    plt.title('GAIL Post-Pretrain Statistical Significance Report')
    plt.xlabel('Discriminator Confidence Score');
    plt.ylabel('Sample Count')
    plt.legend();
    plt.grid(True, alpha=0.2)
    plt.axvline(np.mean(real_scores), color='darkgreen', linestyle='dashed', linewidth=2)
    plt.axvline(np.mean(fake_scores), color='darkred', linestyle='dashed', linewidth=2)
    plt.axvline(np.mean(noise_scores), color='darkgray', linestyle='dashed', linewidth=2)
    plt.text(np.mean(real_scores), plt.ylim()[1] * 0.8, f'Mean Exp: {np.mean(real_scores):.2f}', color='darkgreen')
    plt.text(np.mean(fake_scores), plt.ylim()[1] * 0.7, f'Mean Agent: {np.mean(fake_scores):.2f}', color='darkred')
    plt.text(np.mean(noise_scores), plt.ylim()[1] * 0.8, f'Mean Noise: {np.mean(noise_scores):.2f}', color='darkgray')

    plt.savefig(os.path.join(current_dir, 'analysis_report_pvalue.png'), dpi=300)

    print(f"\n📊 分析完成！报告已保存至 analysis_report_pvalue.png")
    print("-" * 50)
    print(f"🔹 【Agent vs Expert】 JS: {js_fake:.4f} | W-Dist: {w_fake:.4f} | KS-Dist: {ks_fake:.4f}")
    print(f"🔹 【Noise vs Expert】 JS: {js_noise:.4f} | W-Dist: {w_noise:.4f} | KS-Dist: {ks_noise:.4f}")
    print("-" * 50)
    print(f"🧪 统计显著性校验 (基于重采样):")
    print(f"🟢 Agent 相似性 P-value: {p_fake:.4e} " + (
        "(统计学不可区分 - 极佳)" if p_fake > 0.05 else "(存在分布细节差异)"))
    print(f"⚪ Noise 相似性 P-value: {p_noise:.4e} " + (
        "(判别器失效 - 需检查)" if p_noise > 0.05 else "(显著异源 - 达标)"))


if __name__ == '__main__':
    run_standalone_analysis()