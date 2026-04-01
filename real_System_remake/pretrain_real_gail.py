# 作者：杨逸嘉
# 2026年03月11日17时00分35秒
# 内容：
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 1. 完全对齐 TD3.py 的真实生成器 (Actor)
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


# ==========================================
# 2. 完全对齐 gail_core.py 的真实判别器
# ==========================================
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
        x = torch.cat([state, action], dim=1)
        return self.net(x)


# ==========================================
# 3. 标准化模块
# ==========================================
class RunningMeanStd:
    def __init__(self, shape=()):
        self.mean = torch.zeros(shape).to(device)
        self.var = torch.ones(shape).to(device)
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
# 4. 核心预训练逻辑
# ==========================================
def pretrain_gail():
    print(f"=== 🚀 启动真实环境脱机 GAIL 预训练 (设备: {device}) ===")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, 'expert_data_production1_cleaned.csv')

    if not os.path.exists(csv_path):
        print(f"❌ 找不到文件: {csv_path}")
        return

    print("⏳ 正在读取专家数据，请稍候...")
    df = pd.read_csv(csv_path, header=None)
    expert_data = torch.FloatTensor(df.values).to(device)

    total_len = len(expert_data)
    train_size = int(total_len * 0.8)

    train_data = expert_data[:train_size]
    test_data = expert_data[train_size:]

    train_states, train_actions = train_data[:, :33], train_data[:, 33:37]
    test_states, test_actions = test_data[:, :33], test_data[:, 33:37]
    print(f"✅ 数据切分完毕！训练集: {len(train_data)} | 盲测集: {len(test_data)}")

    # 1. 提前计算全局均值和方差，并冻结！
    obs_rms = RunningMeanStd(shape=(33,))
    act_rms = RunningMeanStd(shape=(4,))
    obs_rms.update(train_states)
    act_rms.update(train_actions)
    print("✅ 全局状态与动作标准化统计已完成！")

    # 2. 构造 Mini-Batch DataLoader
    batch_size = 512
    dataset = TensorDataset(train_states, train_actions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    generator = RealActor(s_dim=33, a_dim=4, a_bound=0.5).to(device)
    discriminator = RealDiscriminator(s_dim=33, a_dim=4).to(device)

    opt_G = optim.Adam(generator.parameters(), lr=1e-4)
    opt_D = optim.Adam(discriminator.parameters(), lr=3e-4)

    epochs = 20  # 因为数据量大，20个 Epoch 已经够学很多了
    print(f"\n🧠 开始预训练 (共 {epochs} 轮 Epochs)...")

    for epoch in range(epochs):
        epoch_loss_d, epoch_loss_g = 0, 0

        for batch_s, batch_a in dataloader:
            # 标准化 (带有 clamp 防爆保护)
            s_norm = normalize(batch_s, obs_rms, clamp=True)
            a_norm = normalize(batch_a, act_rms, clamp=False)  # 动作不需要过度 clamp

            # --- A. 训练判别器 ---
            opt_D.zero_grad()
            real_logits = discriminator(s_norm, a_norm)
            loss_D_real = F.binary_cross_entropy_with_logits(real_logits, torch.full_like(real_logits, 0.9))

            with torch.no_grad():
                fake_a_raw = generator(s_norm)
                fake_a_norm = normalize(fake_a_raw, act_rms, clamp=False)

            fake_logits = discriminator(s_norm, fake_a_norm)
            loss_D_fake = F.binary_cross_entropy_with_logits(fake_logits, torch.full_like(fake_logits, 0.1))

            # 随机负样本
            noise_a_raw = (torch.rand_like(batch_a) * 2 - 1) * 0.5
            noise_a_norm = normalize(noise_a_raw, act_rms, clamp=False)
            noise_logits = discriminator(s_norm, noise_a_norm)
            loss_D_noise = F.binary_cross_entropy_with_logits(noise_logits, torch.full_like(noise_logits, 0.1))

            loss_D = loss_D_real + 0.5 * loss_D_fake + 0.5 * loss_D_noise
            loss_D.backward()
            opt_D.step()

            # --- B. 训练生成器 ---
            opt_G.zero_grad()
            gen_a_raw = generator(s_norm)
            gen_a_norm = normalize(gen_a_raw, act_rms, clamp=False)
            gen_logits = discriminator(s_norm, gen_a_norm)

            # 1. GAIL 对抗损失
            loss_gail = F.binary_cross_entropy_with_logits(gen_logits, torch.full_like(gen_logits, 0.9))

            # 2. 基础行为克隆 MSE 损失 (注意：此处 reduction='none' 为了拿到每个元素的细粒度 Loss)
            loss_mse_base = F.mse_loss(gen_a_raw, batch_a, reduction='none')

            # 3. 【终极绝杀】非对称符号暴击判定
            # 找出所有符号相反的位置 (True/False 转化为 1.0/0.0)
            sign_mismatch = (batch_a * gen_a_raw < 0).float()

            # 如果符号错误，在此处的 MSE 乘上 100 倍惩罚！符号正确则保持 1 倍。
            weighted_mse = loss_mse_base * (1.0 + 100.0 * sign_mismatch)

            # 求均值得到最终的监督损失
            loss_mse_final = weighted_mse.mean()

            # 综合损失 (GAIL 权重适度调小，让模型在这个阶段优先听从 MSE 暴击的指挥)
            loss_G = 0.5 * loss_gail + loss_mse_final

            loss_G.backward()
            opt_G.step()

            epoch_loss_d += loss_D.item()
            epoch_loss_g += loss_G.item()

        print(
            f"Epoch {epoch + 1}/{epochs} | Avg Loss D: {epoch_loss_d / len(dataloader):.4f} | Avg Loss G: {epoch_loss_g / len(dataloader):.4f}")

    # ==========================================
    # 5. 盲测评估
    # ==========================================
    print("\n=== 📊 预训练完成，20% 盲测打分 ===")
    discriminator.eval()
    generator.eval()

    with torch.no_grad():
        test_s_norm = normalize(test_states, obs_rms, clamp=True)
        test_a_norm = normalize(test_actions, act_rms, clamp=False)

        # 获取所有测试样本的判别器原始得分 (Sigmoid 后)
        # 转为 CPU 和 numpy 格式以供 matplotlib 使用
        real_scores_raw = torch.sigmoid(discriminator(test_s_norm, test_a_norm)).cpu().numpy().flatten()

        test_fake_raw = generator(test_s_norm)
        test_fake_norm = normalize(test_fake_raw, act_rms, clamp=False)
        fake_scores_raw = torch.sigmoid(discriminator(test_s_norm, test_fake_norm)).cpu().numpy().flatten()

        noise_raw = (torch.rand_like(test_actions) * 2 - 1) * 0.5
        noise_norm = normalize(noise_raw, act_rms, clamp=False)
        noise_scores_raw = torch.sigmoid(discriminator(test_s_norm, noise_norm)).cpu().numpy().flatten()

    # --- 开始绘图逻辑 ---
    plt.figure(figsize=(10, 6))
    # 使用 flatten 后的原始数据绘制直方图
    plt.hist(real_scores_raw, bins=50, alpha=0.6, label='Expert (Real)', color='green', range=(0, 1))
    plt.hist(fake_scores_raw, bins=50, alpha=0.6, label='Agent (Fake)', color='red', range=(0, 1))
    plt.hist(noise_scores_raw, bins=50, alpha=0.3, label='Random Noise', color='gray', range=(0, 1))

    plt.title('GAIL Discriminator Turing Test Results')
    plt.xlabel('Probability (1.0 = Absolutely Expert, 0.0 = Absolutely Fake)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 这里的 current_dir 确保是在脚本同级目录下
    save_fig_path = os.path.join(current_dir, 'gail_test_results.png')
    plt.savefig(save_fig_path, dpi=300)
    print(f"\n🖼️ 图像已保存至: {save_fig_path}")

    # 打印平均分供参考
    print(
        f"🟢 实时均分 - Real: {real_scores_raw.mean():.4f} | Fake: {fake_scores_raw.mean():.4f} | Noise: {noise_scores_raw.mean():.4f}")

    # plt.show()

    # 6. 【整合版】保存所有必要文件
    actor_save_path = os.path.join(current_dir, 'pretrained_actor.pth')
    disc_save_path = os.path.join(current_dir, 'pretrained_discriminator.pth')
    rms_save_path = os.path.join(current_dir, 'obs_rms_params.pth')  # 标准化参数文件

    # 保存网络权重
    torch.save(generator.state_dict(), actor_save_path)
    torch.save(discriminator.state_dict(), disc_save_path)

    # 保存标准化统计量 (生产环境决策必须用到)
    torch.save({
        'mean': obs_rms.mean.cpu(),
        'var': obs_rms.var.cpu(),
        'act_mean': act_rms.mean.cpu(),  # 必须保存！
        'act_var': act_rms.var.cpu()  # 必须保存！
    }, rms_save_path)

    print(f"\n💾 所有文件已保存：")
    print(f" -> 生成器: {actor_save_path}")
    print(f" -> 判别器: {disc_save_path}")
    print(f" -> 标准化参数: {rms_save_path}")


if __name__ == '__main__':
    pretrain_gail()