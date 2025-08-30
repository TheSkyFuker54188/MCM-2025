import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import os
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 创建结果目录
output_dir = "mc_23_A/results/data_analysis"
os.makedirs(output_dir, exist_ok=True)

# 读取数据
data_path = "mc_23_A/data/data_100.csv"
data = pd.read_csv(data_path)

# 提取信用卡通过率和坏账率数据
t_columns = [col for col in data.columns if col.startswith('t_')]
h_columns = [col for col in data.columns if col.startswith('h_')]

# 每行代表一个阈值，共10个阈值
thresholds = range(1, 11)

# 需求1：通过率与坏账率折线图
# 为了便于可视化，选择前10张卡进行画图
def plot_pass_default_rates(num_cards=10):
    plt.figure(figsize=(15, 10))
    
    for i in range(1, num_cards+1):
        # 提取第i张卡的通过率和坏账率
        t_data = data[f't_{i}'].values
        h_data = data[f'h_{i}'].values
        
        plt.plot(t_data, h_data, marker='o', label=f'信用卡{i}')
    
    plt.xlabel('通过率')
    plt.ylabel('坏账率')
    plt.title(f'前{num_cards}张信用卡的通过率与坏账率关系')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/通过率与坏账率折线图(前{num_cards}张卡).png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 所有卡的散点图
    plt.figure(figsize=(12, 8))
    for i in range(1, 101):
        t_data = data[f't_{i}'].values
        h_data = data[f'h_{i}'].values
        plt.scatter(t_data, h_data, s=10, alpha=0.5)
    
    plt.xlabel('通过率')
    plt.ylabel('坏账率')
    plt.title('所有信用卡的通过率与坏账率散点图')
    plt.grid(True)
    plt.savefig(f"{output_dir}/所有信用卡通过率与坏账率散点图.png", dpi=300, bbox_inches='tight')
    plt.close()

# 需求2：计算一百张卡里面的平均通过率与坏账率
def calculate_average_rates():
    # 计算每个阈值下所有卡的平均通过率和坏账率
    average_pass_rates = []
    average_default_rates = []
    
    # 对每个阈值计算平均值
    for row in range(10):
        # 提取该阈值下所有卡的通过率
        t_rates = [data.iloc[row][f't_{i}'] for i in range(1, 101)]
        # 提取该阈值下所有卡的坏账率
        h_rates = [data.iloc[row][f'h_{i}'] for i in range(1, 101)]
        
        # 计算平均值
        avg_t = sum(t_rates) / len(t_rates)
        avg_h = sum(h_rates) / len(h_rates)
        
        average_pass_rates.append(avg_t)
        average_default_rates.append(avg_h)
    
    # 创建结果DataFrame
    result_df = pd.DataFrame({
        '阈值': thresholds,
        '平均通过率': average_pass_rates,
        '平均坏账率': average_default_rates
    })
    
    # 保存到CSV文件
    result_df.to_csv(f"{output_dir}/平均通过率与坏账率.csv", index=False, encoding='utf-8-sig')
    
    # 画出平均通过率和坏账率的折线图
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, average_pass_rates, 'b-o', label='平均通过率')
    plt.grid(True)
    plt.xlabel('阈值')
    plt.ylabel('通过率')
    plt.title('各阈值下的平均通过率')
    plt.savefig(f"{output_dir}/平均通过率.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, average_default_rates, 'r-o', label='平均坏账率')
    plt.grid(True)
    plt.xlabel('阈值')
    plt.ylabel('坏账率')
    plt.title('各阈值下的平均坏账率')
    plt.savefig(f"{output_dir}/平均坏账率.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return result_df

# 需求3：计算出一百张卡里面的最高通过率对应的卡及其坏账率，最高坏账率对应的卡及其通过率
def find_extreme_rates():
    # 对每个阈值分别计算
    results = []
    
    for threshold_idx in range(10):
        # 当前阈值下所有卡的通过率和坏账率
        t_rates = {i: data.iloc[threshold_idx][f't_{i}'] for i in range(1, 101)}
        h_rates = {i: data.iloc[threshold_idx][f'h_{i}'] for i in range(1, 101)}
        
        # 找出最高通过率及对应的卡
        max_pass_card = max(t_rates.items(), key=lambda x: x[1])
        max_pass_card_id = max_pass_card[0]
        max_pass_rate = max_pass_card[1]
        # 该卡的坏账率
        max_pass_card_default = h_rates[max_pass_card_id]
        
        # 找出最高坏账率及对应的卡
        max_default_card = max(h_rates.items(), key=lambda x: x[1])
        max_default_card_id = max_default_card[0]
        max_default_rate = max_default_card[1]
        # 该卡的通过率
        max_default_card_pass = t_rates[max_default_card_id]
        
        results.append({
            '阈值': threshold_idx + 1,
            '最高通过率卡号': max_pass_card_id,
            '最高通过率': max_pass_rate,
            '该卡坏账率': max_pass_card_default,
            '最高坏账率卡号': max_default_card_id,
            '最高坏账率': max_default_rate,
            '该卡通过率': max_default_card_pass
        })
    
    # 创建结果DataFrame
    result_df = pd.DataFrame(results)
    
    # 保存到CSV文件
    result_df.to_csv(f"{output_dir}/极值信用卡分析.csv", index=False, encoding='utf-8-sig')
    
    return result_df

# 需求4：画出所有信用卡的所有通过率与坏账率的箱线图
def plot_boxplots():
    # 提取所有通过率和坏账率
    all_pass_rates = []
    all_default_rates = []
    
    for i in range(1, 101):
        all_pass_rates.extend(data[f't_{i}'].values)
        all_default_rates.extend(data[f'h_{i}'].values)
    
    # 创建箱线图
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.boxplot(all_pass_rates)
    plt.title('所有信用卡通过率箱线图')
    plt.ylabel('通过率')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 2, 2)
    plt.boxplot(all_default_rates)
    plt.title('所有信用卡坏账率箱线图')
    plt.ylabel('坏账率')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/通过率与坏账率箱线图.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 使用seaborn绘制更美观的箱线图
    plt.figure(figsize=(14, 7))
    
    # 创建数据框来存储所有数据点
    boxplot_data = pd.DataFrame({
        '通过率': all_pass_rates,
        '坏账率': all_default_rates
    })
    
    # 使用melt函数重塑数据框以便于使用seaborn
    melted_data = pd.melt(boxplot_data)
    
    # 绘制箱线图
    sns.boxplot(x='variable', y='value', data=melted_data)
    plt.title('通过率与坏账率箱线图')
    plt.xlabel('')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{output_dir}/通过率与坏账率箱线图_seaborn.png", dpi=300, bbox_inches='tight')
    plt.close()

# 需求5：计算通过率与坏账率的spearman相关系数
def calculate_spearman_correlation():
    # 将所有卡片的所有阈值对应的通过率和坏账率数据点收集起来
    all_pass_rates = []
    all_default_rates = []
    
    for i in range(1, 101):
        all_pass_rates.extend(data[f't_{i}'].values)
        all_default_rates.extend(data[f'h_{i}'].values)
    
    # 计算Spearman相关系数
    correlation, p_value = spearmanr(all_pass_rates, all_default_rates)
    
    # 创建散点图，并添加相关系数信息
    plt.figure(figsize=(10, 8))
    plt.scatter(all_pass_rates, all_default_rates, alpha=0.5)
    plt.title(f'通过率与坏账率的相关性 (Spearman相关系数: {correlation:.4f})')
    plt.xlabel('通过率')
    plt.ylabel('坏账率')
    plt.grid(True)
    plt.text(0.05, 0.95, f'Spearman相关系数: {correlation:.4f}\np值: {p_value:.4e}', 
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
    plt.savefig(f"{output_dir}/通过率与坏账率相关性.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return correlation, p_value

def main():
    print("开始数据预处理...")
    
    print("1. 绘制通过率与坏账率折线图...")
    plot_pass_default_rates()
    
    print("2. 计算平均通过率和坏账率...")
    avg_df = calculate_average_rates()
    print(avg_df)
    
    print("3. 查找极值信用卡...")
    extreme_df = find_extreme_rates()
    print(extreme_df)
    
    print("4. 绘制箱线图...")
    plot_boxplots()
    
    print("5. 计算Spearman相关系数...")
    correlation, p_value = calculate_spearman_correlation()
    print(f"通过率与坏账率的Spearman相关系数: {correlation:.4f}, p值: {p_value:.4e}")
    
    print("数据预处理完成，结果保存在 results 目录中。")

if __name__ == "__main__":
    main() 