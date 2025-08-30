import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tabulate import tabulate

def solve_problem1(data_path=None, output_dir=None):
    """
    求解第一问：找出100张信用卡中最优的单张卡片及其阈值设置
    
    参数:
    data_path: 数据文件路径
    output_dir: 输出结果保存目录
    
    返回:
    max_income_card: 最优信用卡编号
    max_income_threshold: 最优阈值编号
    max_income: 最大收入
    """
    print("开始求解第一问...")
    # 根据脚本位置确定项目根目录，支持相对或绝对路径输入
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    if data_path is None:
        data_path = os.path.join(project_root, 'data', 'data_100.csv')
    elif not os.path.isabs(data_path):
        # 相对路径相对于项目根
        data_path = os.path.join(project_root, data_path)

    if output_dir is None:
        output_dir = os.path.join(project_root, 'results', 'problem1')
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
        return None, None, None
    
    # 贷款资金和利率
    M = 1000000  # 贷款资金
    r = 0.08     # 利率
    
    # 创建结果DataFrame
    results = []
    
    # 计算每张卡每个阈值的最终收入
    print("计算所有信用卡在不同阈值下的最终收入...")
    
    # 遍历100张卡
    for card_id in range(1, 101):
        # 提取该卡在不同阈值下的通过率和坏账率
        t_values = data[f't_{card_id}'].values  # 通过率
        h_values = data[f'h_{card_id}'].values  # 坏账率
        
        # 计算不同阈值下的收入
        for threshold_idx in range(10):
            t = t_values[threshold_idx]  # 通过率
            h = h_values[threshold_idx]  # 坏账率
            
            # 根据公式计算最终收入：I_i^j = 1000000 × t_i^j × (0.08-1.08h_i^j)
            income = M * t * (r - (1 + r) * h)
            
            # 将结果添加到列表
            results.append({
                '信用卡编号': card_id,
                '阈值编号': threshold_idx + 1,
                '通过率': t,
                '坏账率': h,
                '最终收入': income
            })
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 保存全部结果
    results_df.to_csv(f"{output_dir}/problem1_all_results.csv", index=False, encoding='utf-8-sig')
    
    # 找出最大收入及其对应的卡片和阈值
    max_income_row = results_df.loc[results_df['最终收入'].idxmax()]
    max_income_card = int(max_income_row['信用卡编号'])
    max_income_threshold = int(max_income_row['阈值编号'])
    max_income = max_income_row['最终收入']
    
    # 输出最优结果
    print("\n最优结果:")
    print(f"最优信用卡编号: {max_income_card}")
    print(f"最优阈值编号: {max_income_threshold}")
    print(f"最大收入: {max_income:.2f}元")
    
    # 将最优结果保存到文件
    with open(f"{output_dir}/problem1_best_result.txt", 'w', encoding='utf-8') as f:
        f.write(f"最优信用卡编号: {max_income_card}\n")
        f.write(f"最优阈值编号: {max_income_threshold}\n")
        f.write(f"最大收入: {max_income:.2f}元\n")
        f.write(f"对应通过率: {max_income_row['通过率']:.4f}\n")
        f.write(f"对应坏账率: {max_income_row['坏账率']:.4f}\n")
    
    # 绘制最优卡片的收入图表
    optimal_card_results = results_df[results_df['信用卡编号'] == max_income_card]
    plt.figure(figsize=(10, 6))
    plt.plot(optimal_card_results['阈值编号'], optimal_card_results['最终收入'], 'bo-', linewidth=2)
    plt.axvline(x=max_income_threshold, color='r', linestyle='--', label=f'最优阈值 ({max_income_threshold})')
    plt.xlabel('阈值编号')
    plt.ylabel('最终收入 (元)')
    plt.title(f'信用卡 {max_income_card} 在不同阈值下的收入')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/problem1_optimal_card_income.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 返回最优结果
    return max_income_card, max_income_threshold, max_income

if __name__ == "__main__":
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    
    # 求解第一问（使用基于脚本位置的默认路径）
    max_income_card, max_income_threshold, max_income = solve_problem1()
    
    # 打印结果表格
    if max_income is not None:
        result_table = [
            ["最优信用卡编号", max_income_card],
            ["最优阈值编号", max_income_threshold],
            ["最大收入", f"{max_income:.2f}元"]
        ]
        print("\n最终结果:")
        print(tabulate(result_table, tablefmt="grid")) 