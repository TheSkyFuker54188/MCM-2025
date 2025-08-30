import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tabulate import tabulate
import dimod  # D-Wave的QUBO求解器
from neal import SimulatedAnnealingSampler  # 量子模拟退火求解器

def solve_problem1_qubo(data_path=None, output_dir=None):
    """
    使用QUBO模型求解第一问：找出100张信用卡中最优的单张卡片及其阈值设置
    
    参数:
    data_path: 数据文件路径
    output_dir: 输出结果保存目录
    
    返回:
    max_income_card: 最优信用卡编号
    max_income_threshold: 最优阈值编号
    max_income: 最大收入
    """
    print("开始使用QUBO模型求解第一问...")
    
    # 根据脚本位置确定项目根目录，支持相对或绝对路径输入
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    if data_path is None:
        data_path = os.path.join(project_root, 'data', 'data_100.csv')
    elif not os.path.isabs(data_path):
        # 相对路径相对于项目根
        data_path = os.path.join(project_root, data_path)

    if output_dir is None:
        output_dir = os.path.join(project_root, 'results', 'problem1_qubo')
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
    
    # 计算所有信用卡在不同阈值下的最终收入
    print("计算所有信用卡在不同阈值下的最终收入并构建QUBO模型...")
    
    # 构建QUBO矩阵
    Q = {}
    # 记录每个决策变量对应的卡号和阈值
    var_mapping = {}
    # 记录每个卡号和阈值对应的变量索引
    idx_mapping = {}
    # 记录每个决策变量对应的收入
    incomes = {}
    
    # 变量索引
    idx = 0
    
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
            
            # 记录变量映射
            var_mapping[idx] = (card_id, threshold_idx + 1)
            idx_mapping[(card_id, threshold_idx + 1)] = idx
            
            # 记录收入
            incomes[idx] = income
            
            # 添加线性项到QUBO矩阵（目标是最小化负收入，等价于最大化收入）
            Q[(idx, idx)] = -income
            
            idx += 1
    
    # 添加约束条件：只选一张卡的一个阈值
    A = 1e6  # 罚函数系数
    
    # 添加约束条件的二次项
    for i in range(idx):
        for j in range(idx):
            if i != j:
                Q[(i, j)] = Q.get((i, j), 0) + A
    
    # 添加约束条件的线性项
    for i in range(idx):
        Q[(i, i)] = Q.get((i, i), 0) - 2 * A
    
    # 添加常数项（在QUBO求解中不影响结果，但影响最终的目标函数值）
    # 注意：常数项在dimod中通常使用offset参数设置，这里我们不需要显式添加
    
    # 转换为BinaryQuadraticModel
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    
    # 使用模拟退火求解器
    print("使用量子模拟退火算法求解QUBO模型...")
    sampler = SimulatedAnnealingSampler()
    response = sampler.sample(bqm, num_reads=1000)  #! 进行1000次采样
    
    # 获取最佳解
    best_solution = response.first.sample
    
    # 找出值为1的变量
    selected_var = None
    for var, val in best_solution.items():
        if val == 1:
            selected_var = var
            break
    
    if selected_var is not None:
        # 获取对应的卡号和阈值
        max_income_card, max_income_threshold = var_mapping[selected_var]
        max_income = incomes[selected_var]
        
        # 检查结果是否正确
        t = data[f't_{max_income_card}'].values[max_income_threshold - 1]
        h = data[f'h_{max_income_card}'].values[max_income_threshold - 1]
        calculated_income = M * t * (r - (1 + r) * h)
        
        # 验证计算结果
        assert abs(max_income - calculated_income) < 1e-8, "计算结果与模型结果不一致"
    else:
        print("求解失败，未找到有效解")
        return None, None, None
    
    # 为验证QUBO结果，也计算所有可能的收入并找出最大值
    all_results = []
    for card_id in range(1, 101):
        t_values = data[f't_{card_id}'].values
        h_values = data[f'h_{card_id}'].values
        
        for threshold_idx in range(10):
            t = t_values[threshold_idx]
            h = h_values[threshold_idx]
            income = M * t * (r - (1 + r) * h)
            
            all_results.append({
                '信用卡编号': card_id,
                '阈值编号': threshold_idx + 1,
                '通过率': t,
                '坏账率': h,
                '最终收入': income
            })
    
    # 转换为DataFrame并保存
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"{output_dir}/problem1_all_results.csv", index=False, encoding='utf-8-sig')
    
    # 找出最大收入（用于验证QUBO结果）
    max_income_row = results_df.loc[results_df['最终收入'].idxmax()]
    brute_force_card = int(max_income_row['信用卡编号'])
    brute_force_threshold = int(max_income_row['阈值编号'])
    brute_force_income = max_income_row['最终收入']
    
    # 输出最优结果
    print("\nQUBO求解结果:")
    print(f"最优信用卡编号: {max_income_card}")
    print(f"最优阈值编号: {max_income_threshold}")
    print(f"最大收入: {max_income:.2f}元")
    
    print("\n暴力枚举验证结果:")
    print(f"最优信用卡编号: {brute_force_card}")
    print(f"最优阈值编号: {brute_force_threshold}")
    print(f"最大收入: {brute_force_income:.2f}元")
    
    # 验证两种方法结果是否一致
    if max_income_card == brute_force_card and max_income_threshold == brute_force_threshold:
        print("\n✓ QUBO结果与暴力枚举结果一致")
    else:
        print("\n✗ QUBO结果与暴力枚举结果不一致，请检查QUBO模型")
    
    # 将最优结果保存到文件
    with open(f"{output_dir}/problem1_best_result.txt", 'w', encoding='utf-8') as f:
        f.write(f"QUBO求解结果:\n")
        f.write(f"最优信用卡编号: {max_income_card}\n")
        f.write(f"最优阈值编号: {max_income_threshold}\n")
        f.write(f"最大收入: {max_income:.2f}元\n")
        f.write(f"对应通过率: {data[f't_{max_income_card}'].values[max_income_threshold - 1]:.4f}\n")
        f.write(f"对应坏账率: {data[f'h_{max_income_card}'].values[max_income_threshold - 1]:.4f}\n\n")
        
        f.write(f"暴力枚举验证结果:\n")
        f.write(f"最优信用卡编号: {brute_force_card}\n")
        f.write(f"最优阈值编号: {brute_force_threshold}\n")
        f.write(f"最大收入: {brute_force_income:.2f}元\n")
    
    # 绘制最优卡片的收入图表
    optimal_card_results = results_df[results_df['信用卡编号'] == max_income_card]
    plt.figure(figsize=(10, 6))
    plt.plot(optimal_card_results['阈值编号'], optimal_card_results['最终收入'], 'bo-', linewidth=2)
    plt.axvline(x=max_income_threshold, color='r', linestyle='--', label=f'最优阈值 ({max_income_threshold})')
    plt.xlabel('阈值编号')
    plt.ylabel('最终收入 (元)')
    plt.title(f'信用卡 {max_income_card} 在不同阈值下的收入 (QUBO求解)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/problem1_optimal_card_income.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 返回最优结果
    return max_income_card, max_income_threshold, max_income

def visualize_qubo_matrix(Q, output_dir):
    """
    可视化QUBO矩阵
    """
    # 确定矩阵维度
    max_idx = max(max(i, j) for i, j in Q.keys()) + 1
    
    # 创建矩阵
    qubo_matrix = np.zeros((max_idx, max_idx))
    
    # 填充矩阵
    for (i, j), val in Q.items():
        qubo_matrix[i, j] = val
    
    # 绘制热图
    plt.figure(figsize=(12, 10))
    plt.imshow(qubo_matrix, cmap='viridis')
    plt.colorbar(label='QUBO系数')
    plt.title('QUBO矩阵可视化')
    plt.xlabel('变量索引')
    plt.ylabel('变量索引')
    plt.savefig(f"{output_dir}/qubo_matrix_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    
    # 求解第一问（使用基于脚本位置的默认路径）
    max_income_card, max_income_threshold, max_income = solve_problem1_qubo()
    
    # 打印结果表格
    if max_income is not None:
        result_table = [
            ["最优信用卡编号", max_income_card],
            ["最优阈值编号", max_income_threshold],
            ["最大收入", f"{max_income:.2f}元"]
        ]
        print("\n最终结果:")
        print(tabulate(result_table, tablefmt="grid")) 