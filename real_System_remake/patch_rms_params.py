# 作者：杨逸嘉
# 2026年03月23日21时49分10秒
# 内容：
import torch
import pandas as pd
import numpy as np
import os


def patch_params():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rms_path = os.path.join(current_dir, 'obs_rms_params.pth')
    csv_path = os.path.join(current_dir, 'expert_data_production1_cleaned.csv')

    if not os.path.exists(rms_path) or not os.path.exists(csv_path):
        print("❌ 找不到 .pth 文件或专家 .csv 文件，请检查路径。")
        return

    # 1. 加载现有的 obs 参数
    params = torch.load(rms_path)
    print(f"📂 已读取现有参数，包含键: {list(params.keys())}")

    # 2. 从 CSV 计算动作的均值和方差 (假设动作在 33-37 列)
    print("⏳ 正在计算专家动作的标准化指标...")
    df = pd.read_csv(csv_path, header=None)
    # 转换为 numpy 提取动作列 (第 33 到 36 列)
    expert_actions = df.values[:, 33:37]

    act_mean = np.mean(expert_actions, axis=0)
    act_var = np.var(expert_actions, axis=0)

    # 3. 注入新参数
    params['act_mean'] = torch.FloatTensor(act_mean)
    params['act_var'] = torch.FloatTensor(act_var)

    # 4. 覆盖保存
    torch.save(params, rms_path)
    print("✅ 补丁成功！'act_mean' 和 'act_var' 已存入 obs_rms_params.pth")
    print(f"📊 动作均值: {act_mean}")


if __name__ == '__main__':
    patch_params()