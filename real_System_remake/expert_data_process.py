import pandas as pd
import numpy as np
import os


def clean_single_file(input_path, output_path):
    print(f"正在读取文件: {input_path} ...")

    try:
        # 1. 读取数据（因为没有表头，加上 header=None）
        df = pd.read_csv(input_path, header=None)
        first_col = df.columns[0]

        # 为了防止列中有非数字字符导致报错，先强制转换为数字
        col_numeric = pd.to_numeric(df[first_col], errors='coerce')

        # 2. 找到所有 0.1 的位置作为切分标记
        is_marker = (col_numeric.round(5) == 0.1)

        # 3. 使用 cumsum() 给每段数据生成独立的“组号”
        group_id = is_marker.cumsum()

        # 4. 统计每一组的行数（间隔数）
        group_sizes = df.groupby(group_id).size()

        # 5. 核心过滤：找出所有行数 >= 90 的组号
        valid_groups = group_sizes[group_sizes >= 90].index

        # 按照合法的组号过滤原表格
        filtered_df = df[group_id.isin(valid_groups)]

        # 6. 保存处理后的数据（保持无表头格式）
        filtered_df.to_csv(output_path, index=False, header=False, encoding='utf-8-sig')

        # 打印清洗效果报告
        print(f"\n✅ 清洗成功！")
        print(f"📊 数据对比: 原本 {len(df)} 行 -> 过滤后剩余 {len(filtered_df)} 行")
        print(f"🗑️ 共删除了 {len(df) - len(filtered_df)} 行（即那些间隔小于90的片段）")
        print(f"📁 结果已保存至: {output_path}")

    except FileNotFoundError:
        print("❌ 找不到文件！请检查下方的 file_path 路径是否拼写正确。")
    except Exception as e:
        print(f"❌ 处理失败，错误信息: {e}")


# ==========================================
# 你的原文件路径
# ==========================================
file_path = r"D:\福州大学\3.课外\实验\经济系统clean\real_System_remake\expert_data_production1.csv"

# 清洗后生成的新文件路径（加了 _cleaned 后缀）
output_path = r"D:\福州大学\3.课外\实验\经济系统clean\real_System_remake\expert_data_production1_cleaned.csv"

# 运行函数
clean_single_file(file_path, output_path)