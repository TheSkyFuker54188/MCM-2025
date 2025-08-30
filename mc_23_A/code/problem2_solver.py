import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tabulate import tabulate
import itertools

def solve_problem2(data_path=None, output_dir=None):
    """
    求解第二问：找出给定的3张信用卡的最优阈值组合
    
    参数:
    data_path: 数据文件路径
    output_dir: 输出结果保存目录
    
    返回:
    best_thresholds: 最优阈值组合 [阈值1, 阈值2, 阈值3]
    max_income: 最大收入
    """
    print("开始求解第二问...")
    
    # 根据脚本位置确定项目根目录，支持相对或绝对路径输入
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    if data_path is None:
        data_path = os.path.join(project_root, 'data', 'data_100.csv')
    elif not os.path.isabs(data_path):
        # 相对路径相对于项目根
        data_path = os.path.join(project_root, data_path)

    if output_dir is None:
        output_dir = os.path.join(project_root, 'results', 'problem2')
    elif not os.path.isabs(output_dir):
        output_dir = os.path.join(project_root, output_dir)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据
    try:
        data = pd.read_csv(data_path)
        print(f"成功读取数据文件: {data_path}")
    except Exception as e:
        print(f"读取数据文件失败: {e}")
        return None, None
    
    # 贷款资金和利率
    M = 1000000  # 贷款资金
    r = 0.08     # 利率
    
    # 提取三张卡的通过率和坏账率数据
    # 注意：题目要求使用卡号1, 2, 3
    cards = [1, 2, 3]
    thresholds = range(1, 11)  # 10个阈值
    
    # 存储所有卡片在不同阈值下的通过率和坏账率
    t_rates = {}  # 通过率
    h_rates = {}  # 坏账率
    
    for card_id in cards:
        t_rates[card_id] = data[f't_{card_id}'].values
        h_rates[card_id] = data[f'h_{card_id}'].values
    
    print("开始暴力搜索最优阈值组合...")
    
    # 存储所有组合的收入结果
    results = []
    
    # 遍历所有可能的阈值组合
    for i, j, k in itertools.product(range(10), repeat=3):
        # 计算总平均通过率和总平均坏账率
        t1 = t_rates[1][i]  # 第1张卡第i个阈值的通过率
        t2 = t_rates[2][j]  # 第2张卡第j个阈值的通过率
        t3 = t_rates[3][k]  # 第3张卡第k个阈值的通过率
        
        h1 = h_rates[1][i]  # 第1张卡第i个阈值的坏账率
        h2 = h_rates[2][j]  # 第2张卡第j个阈值的坏账率
        h3 = h_rates[3][k]  # 第3张卡第k个阈值的坏账率
        
        # 计算总平均通过率和总平均坏账率
        T = (t1 + t2 + t3) / 3
        H = (h1 + h2 + h3) / 3
        
        # 计算最终收入: I = M × T × (r - (1 + r) × H)
        income = M * T * (r - (1 + r) * H)
        
        # 将结果添加到列表
        results.append({
            '阈值1': i + 1,
            '阈值2': j + 1,
            '阈值3': k + 1,
            '通过率1': t1,
            '通过率2': t2,
            '通过率3': t3,
            '坏账率1': h1,
            '坏账率2': h2,
            '坏账率3': h3,
            '总平均通过率': T,
            '总平均坏账率': H,
            '最终收入': income
        })
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 保存所有结果
    results_df.to_csv(f"{output_dir}/problem2_all_results.csv", index=False, encoding='utf-8-sig')
    
    # 找出最大收入及其对应的阈值组合
    max_income_row = results_df.loc[results_df['最终收入'].idxmax()]
    best_thresholds = [int(max_income_row['阈值1']), int(max_income_row['阈值2']), int(max_income_row['阈值3'])]
    max_income = max_income_row['最终收入']
    
    # 输出最优结果
    print("\n最优结果:")
    print(f"最优阈值组合: [{best_thresholds[0]}, {best_thresholds[1]}, {best_thresholds[2]}]")
    print(f"最大收入: {max_income:.2f}元")
    
    # 将最优结果保存到文件
    with open(f"{output_dir}/problem2_best_result.txt", 'w', encoding='utf-8-sig') as f:
        f.write(f"最优阈值组合: [{best_thresholds[0]}, {best_thresholds[1]}, {best_thresholds[2]}]\n")
        f.write(f"最大收入: {max_income:.2f}元\n")
        f.write(f"各卡通过率: [{max_income_row['通过率1']:.4f}, {max_income_row['通过率2']:.4f}, {max_income_row['通过率3']:.4f}]\n")
        f.write(f"各卡坏账率: [{max_income_row['坏账率1']:.4f}, {max_income_row['坏账率2']:.4f}, {max_income_row['坏账率3']:.4f}]\n")
        f.write(f"总平均通过率: {max_income_row['总平均通过率']:.4f}\n")
        f.write(f"总平均坏账率: {max_income_row['总平均坏账率']:.4f}\n")
    
    # 绘制收入热力图 (以阈值1和阈值2为坐标，固定最优阈值3)
    best_threshold3 = best_thresholds[2]
    pivot_data = results_df[results_df['阈值3'] == best_threshold3].pivot(
        index='阈值1', columns='阈值2', values='最终收入')
    
    plt.figure(figsize=(10, 8))
    sns_heatmap = plt.imshow(pivot_data.values, cmap='viridis')
    plt.colorbar(sns_heatmap, label='最终收入 (元)')
    plt.title(f'固定阈值3={best_threshold3}时，不同阈值1和阈值2组合的收入热力图')
    plt.xlabel('阈值2')
    plt.ylabel('阈值1')
    plt.xticks(range(10), range(1, 11))
    plt.yticks(range(10), range(1, 11))
    
    # 标记最优点
    if best_thresholds[2] == best_threshold3:
        plt.plot(best_thresholds[1]-1, best_thresholds[0]-1, 'r*', markersize=15, label='最优组合')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/problem2_income_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制3D散点图，显示所有组合的收入
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 抽样展示部分数据点，避免图像过于拥挤
    sample_size = 100
    sampled_results = results_df.sample(sample_size) if len(results_df) > sample_size else results_df
    
    scatter = ax.scatter(sampled_results['阈值1'], sampled_results['阈值2'], sampled_results['阈值3'],
                         c=sampled_results['最终收入'], cmap='viridis', s=30, alpha=0.7)
    
    # 标记最优点
    ax.scatter([best_thresholds[0]], [best_thresholds[1]], [best_thresholds[2]], 
               color='red', s=100, marker='*', label='最优组合')
    
    ax.set_xlabel('阈值1')
    ax.set_ylabel('阈值2')
    ax.set_zlabel('阈值3')
    ax.set_title('不同阈值组合的收入分布 (抽样展示)')
    
    # 设置坐标轴刻度
    ax.set_xticks(range(1, 11))
    ax.set_yticks(range(1, 11))
    ax.set_zticks(range(1, 11))
    
    plt.colorbar(scatter, ax=ax, label='最终收入 (元)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/problem2_3d_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return best_thresholds, max_income

if __name__ == "__main__":
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    
    # 添加seaborn库用于更美观的可视化
    try:
        import seaborn as sns
        sns.set_style("whitegrid")  # 设置绘图风格
    except ImportError:
        print("警告: seaborn库未安装，将使用默认matplotlib风格")
    
    # 求解第二问
    best_thresholds, max_income = solve_problem2()
    
    # 打印结果表格
    if max_income is not None:
        result_table = [
            ["最优阈值组合", f"[{best_thresholds[0]}, {best_thresholds[1]}, {best_thresholds[2]}]"],
            ["最大收入", f"{max_income:.2f}元"]
        ]
        print("\n最终结果:")
        print(tabulate(result_table, tablefmt="grid")) 