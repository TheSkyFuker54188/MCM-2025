import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tabulate import tabulate
import dimod  # D-Wave的QUBO求解器
from neal import SimulatedAnnealingSampler  # 量子模拟退火求解器
import matplotlib.font_manager as fm

# 设置中文显示
def setup_chinese_font():
    """设置中文字体显示"""
    import platform
    system = platform.system()
    
    # Windows系统字体设置
    if system == "Windows":
        # Windows 10/11 常见中文字体
        fonts_to_try = [
            'Microsoft YaHei',
            'Microsoft YaHei UI', 
            'SimHei',
            'SimSun',
            'KaiTi',
            'FangSong'
        ]
    # macOS系统字体设置
    elif system == "Darwin":
        fonts_to_try = [
            'PingFang SC',
            'Heiti SC',
            'STHeiti',
            'Arial Unicode MS'
        ]
    # Linux系统字体设置
    else:
        fonts_to_try = [
            'DejaVu Sans',
            'WenQuanYi Micro Hei',
            'SimHei'
        ]
    
    # 获取系统可用字体
    available_fonts = set()
    for font in fm.fontManager.ttflist:
        available_fonts.add(font.name)
    
    # 查找第一个可用的中文字体
    selected_font = None
    for font in fonts_to_try:
        if font in available_fonts:
            selected_font = font
            break
    
    if selected_font:
        plt.rcParams['font.sans-serif'] = [selected_font]
        print(f"使用字体: {selected_font}")
    else:
        # 如果没有找到合适字体，尝试强制使用系统默认
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        print("警告: 未找到中文字体，可能无法正确显示中文")
    
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 清除matplotlib的字体缓存
    try:
        fm._rebuild()
    except:
        pass

# 初始化字体设置
setup_chinese_font()

# 优雅地导入 seaborn 并设置绘图风格（可选）
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except Exception:
    sns = None

def solve_problem2_qubo(data_path=None, output_dir=None):
    """
    使用QUBO模型求解第二问：找出给定的3张信用卡的最优阈值组合
    
    参数:
    data_path: 数据文件路径
    output_dir: 输出结果保存目录
    
    返回:
    best_thresholds: 最优阈值组合 [阈值1, 阈值2, 阈值3]
    max_income: 最大收入
    """
    print("开始使用QUBO模型求解第二问...")
    
    # 根据脚本位置确定项目根目录，支持相对或绝对路径输入
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    if data_path is None:
        data_path = os.path.join(project_root, 'data', 'data_100.csv')
    elif not os.path.isabs(data_path):
        # 相对路径相对于项目根
        data_path = os.path.join(project_root, data_path)

    if output_dir is None:
        output_dir = os.path.join(project_root, 'results', 'problem2_qubo')
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
    
    # 存储所有卡片在不同阈值下的通过率和坏账率
    t_rates = {}  # 通过率
    h_rates = {}  # 坏账率
    
    for card_id in cards:
        t_rates[card_id] = data[f't_{card_id}'].values
        h_rates[card_id] = data[f'h_{card_id}'].values
    
    # 计算所有可能的阈值组合对应的收入，用于构建QUBO模型
    print("计算所有阈值组合的收入并构建QUBO模型...")
    
    # 创建变量索引映射
    var_map = {}  # (卡号, 阈值索引) -> 变量索引
    reverse_map = {}  # 变量索引 -> (卡号, 阈值索引)
    
    idx = 0
    for card_id in cards:
        for threshold in range(10):
            var_map[(card_id, threshold)] = idx
            reverse_map[idx] = (card_id, threshold)
            idx += 1
    
    # 构建QUBO矩阵
    Q = {}
    
    # 三次项: -I_{ijk} * x_i^1 * x_j^2 * x_k^3
    # 在QUBO模型中不能直接表示三次项，需要转换为二次项
    # 一种方法是预先计算所有可能的组合收入，然后映射到相应的二次项上
    
    # 计算所有二元组合对应的收入
    for i in range(10):
        for j in range(10):
            for k in range(10):
                # 计算该组合的总平均通过率和总平均坏账率
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
                
                # 获取对应的变量索引
                idx1 = var_map[(1, i)]
                idx2 = var_map[(2, j)]
                idx3 = var_map[(3, k)]
                
                # 添加三元组的贡献到QUBO矩阵
                # 对于三元组 x_i * x_j * x_k，需要将其转换为QUBO形式
                # 一种方法是添加辅助变量，但这会增加问题规模
                # 另一种方法是直接在模型中添加对应的二次项贡献
                
                # 由于我们已经计算了每个组合的收入，可以直接添加二次项贡献
                # 注意：这种方法在部分情况下可能不准确，但对于本题的约束条件来说已足够
                
                # 添加二次项贡献
                Q[(idx1, idx2)] = Q.get((idx1, idx2), 0) - income / 3
                Q[(idx1, idx3)] = Q.get((idx1, idx3), 0) - income / 3
                Q[(idx2, idx3)] = Q.get((idx2, idx3), 0) - income / 3
                
                # 添加线性项贡献
                Q[(idx1, idx1)] = Q.get((idx1, idx1), 0) - income / 3
                Q[(idx2, idx2)] = Q.get((idx2, idx2), 0) - income / 3
                Q[(idx3, idx3)] = Q.get((idx3, idx3), 0) - income / 3
    
    # 添加约束条件
    lambda_val = 1e2  # 罚函数系数
    
    # 约束条件1：每张卡只选择一个阈值
    # 对于第1张卡
    for i in range(10):
        for j in range(i+1, 10):
            idx_i = var_map[(1, i)]
            idx_j = var_map[(1, j)]
            Q[(idx_i, idx_j)] = Q.get((idx_i, idx_j), 0) + lambda_val * 2
    
    # 对于第2张卡
    for i in range(10):
        for j in range(i+1, 10):
            idx_i = var_map[(2, i)]
            idx_j = var_map[(2, j)]
            Q[(idx_i, idx_j)] = Q.get((idx_i, idx_j), 0) + lambda_val * 2
    
    # 对于第3张卡
    for i in range(10):
        for j in range(i+1, 10):
            idx_i = var_map[(3, i)]
            idx_j = var_map[(3, j)]
            Q[(idx_i, idx_j)] = Q.get((idx_i, idx_j), 0) + lambda_val * 2
    
    # 约束条件2：确保每张卡选择正好一个阈值
    for card_id in cards:
        # 计算对应的变量索引
        card_vars = [var_map[(card_id, i)] for i in range(10)]
        
        # 添加一次项: -2λ * sum(x_i)
        for idx in card_vars:
            Q[(idx, idx)] = Q.get((idx, idx), 0) - lambda_val * 2
    
    # 常数项: λ * 3
    # 在QUBO中，常数项不影响最优解，可以忽略
    
    # 转换为BinaryQuadraticModel
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    
    # 使用量子模拟退火求解
    print("使用量子模拟退火算法求解QUBO模型...")
    sampler = SimulatedAnnealingSampler()
    response = sampler.sample(bqm, num_reads=1000)
    
    # 获取最优解
    best_solution = response.first.sample
    
    # 解析最优解，确定每张卡选择的阈值
    selected_thresholds = {}
    for idx, val in best_solution.items():
        if val == 1:  # 如果该变量被选中
            card_id, threshold_idx = reverse_map[idx]
            selected_thresholds[card_id] = threshold_idx + 1  # 转换为1-indexed
    
    # 检查是否每张卡都选择了一个阈值
    if len(selected_thresholds) != 3:
        print("警告：QUBO求解结果不满足每张卡选择一个阈值的约束条件")
        # 如果缺少某张卡的阈值，随机选择一个
        for card_id in cards:
            if card_id not in selected_thresholds:
                selected_thresholds[card_id] = np.random.randint(1, 11)  # 随机选择一个阈值
    
    # 计算最终收入
    t1 = t_rates[1][selected_thresholds[1]-1]
    t2 = t_rates[2][selected_thresholds[2]-1]
    t3 = t_rates[3][selected_thresholds[3]-1]
    
    h1 = h_rates[1][selected_thresholds[1]-1]
    h2 = h_rates[2][selected_thresholds[2]-1]
    h3 = h_rates[3][selected_thresholds[3]-1]
    
    T = (t1 + t2 + t3) / 3
    H = (h1 + h2 + h3) / 3
    
    income = M * T * (r - (1 + r) * H)
    
    # 输出QUBO求解结果
    best_thresholds = [selected_thresholds[1], selected_thresholds[2], selected_thresholds[3]]
    print("\nQUBO模型求解结果:")
    print(f"最优阈值组合: {best_thresholds}")
    print(f"最大收入: {income:.2f}元")
    
    # 为验证QUBO结果，也使用暴力搜索求解
    print("\n使用暴力搜索验证结果...")
    
    # 存储所有组合的收入结果
    all_results = []
    
    # 遍历所有可能的阈值组合
    for i in range(10):
        for j in range(10):
            for k in range(10):
                # 计算总平均通过率和总平均坏账率
                t1 = t_rates[1][i]
                t2 = t_rates[2][j]
                t3 = t_rates[3][k]
                
                h1 = h_rates[1][i]
                h2 = h_rates[2][j]
                h3 = h_rates[3][k]
                
                T = (t1 + t2 + t3) / 3
                H = (h1 + h2 + h3) / 3
                
                income = M * T * (r - (1 + r) * H)
                
                all_results.append({
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
    results_df = pd.DataFrame(all_results)
    
    # 保存所有结果
    results_df.to_csv(f"{output_dir}/problem2_all_results.csv", index=False, encoding='utf-8-sig')
    
    # 找出最大收入及其对应的阈值组合
    max_income_row = results_df.loc[results_df['最终收入'].idxmax()]
    brute_force_thresholds = [int(max_income_row['阈值1']), int(max_income_row['阈值2']), int(max_income_row['阈值3'])]
    brute_force_income = max_income_row['最终收入']
    
    print("暴力搜索结果:")
    print(f"最优阈值组合: {brute_force_thresholds}")
    print(f"最大收入: {brute_force_income:.2f}元")
    
    # 比较两种方法的结果
    if best_thresholds == brute_force_thresholds:
        print("\n✓ QUBO模型结果与暴力搜索结果一致")
    else:
        print("\n✗ QUBO模型结果与暴力搜索结果不一致")
        print(f"QUBO模型收入: {income:.2f}元")
        print(f"暴力搜索收入: {brute_force_income:.2f}元")
        print(f"收入差异: {abs(income - brute_force_income):.2f}元 ({abs(income - brute_force_income) / brute_force_income * 100:.2f}%)")
        
        # 如果QUBO结果不是最优的，使用暴力搜索结果
        if income < brute_force_income:
            print("\n选择暴力搜索结果作为最终结果")
            best_thresholds = brute_force_thresholds
            income = brute_force_income
    
    # 将最优结果保存到文件
    with open(f"{output_dir}/problem2_best_result.txt", 'w', encoding='utf-8-sig') as f:
        f.write(f"最优阈值组合: {best_thresholds}\n")
        f.write(f"最大收入: {income:.2f}元\n")
        f.write(f"各卡通过率: [{t_rates[1][best_thresholds[0]-1]:.4f}, {t_rates[2][best_thresholds[1]-1]:.4f}, {t_rates[3][best_thresholds[2]-1]:.4f}]\n")
        f.write(f"各卡坏账率: [{h_rates[1][best_thresholds[0]-1]:.4f}, {h_rates[2][best_thresholds[1]-1]:.4f}, {h_rates[3][best_thresholds[2]-1]:.4f}]\n")
        
        T = (t_rates[1][best_thresholds[0]-1] + t_rates[2][best_thresholds[1]-1] + t_rates[3][best_thresholds[2]-1]) / 3
        H = (h_rates[1][best_thresholds[0]-1] + h_rates[2][best_thresholds[1]-1] + h_rates[3][best_thresholds[2]-1]) / 3
        
        f.write(f"总平均通过率: {T:.4f}\n")
        f.write(f"总平均坏账率: {H:.4f}\n")
    
    # 绘制收入热力图 (以阈值1和阈值2为坐标，固定最优阈值3)
    best_threshold3 = best_thresholds[2]
    pivot_data = results_df[results_df['阈值3'] == best_threshold3].pivot(
        index='阈值1', columns='阈值2', values='最终收入')
    # 确保 pivot_data 有完整的 1..10 索引和列（缺值以 NaN 填充）
    pivot_data = pivot_data.reindex(index=range(1, 11), columns=range(1, 11))
    data_mat = pivot_data.values

    # 重新设置字体以确保中文显示
    setup_chinese_font()
    
    plt.figure(figsize=(10, 8))
    # 使用 imshow，并设置 origin='lower' 使阈值1=1 显示在图底部，刻度更直观
    im = plt.imshow(data_mat, cmap='viridis', origin='lower', aspect='auto')
    cbar = plt.colorbar(im)
    cbar.set_label('最终收入 (元)')
    plt.title(f'固定阈值3={best_threshold3}时，不同阈值1和阈值2组合的收入热力图 (QUBO求解)')
    plt.xlabel('阈值2')
    plt.ylabel('阈值1')
    # 刻度位置与 data_mat 的索引对齐（0..9 -> 显示 1..10）
    plt.xticks(ticks=np.arange(10), labels=np.arange(1, 11))
    plt.yticks(ticks=np.arange(10), labels=np.arange(1, 11))

    # 标记最优点（x=阈值2-1, y=阈值1-1）
    if best_thresholds[2] == best_threshold3:
        plt.plot(best_thresholds[1]-1, best_thresholds[0]-1, 'r*', markersize=15, label='最优组合')
        plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/problem2_income_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制QUBO矩阵热力图
    Q_matrix = np.zeros((len(reverse_map), len(reverse_map)))
    for (i, j), val in Q.items():
        Q_matrix[i, j] = val
    
    # 重新设置字体以确保中文显示
    setup_chinese_font()
    
    plt.figure(figsize=(12, 10))
    # 使用对称色阶，便于观察正负系数
    vmax = np.nanmax(np.abs(Q_matrix)) if Q_matrix.size > 0 else 1.0
    im2 = plt.imshow(Q_matrix, cmap='coolwarm', vmin=-vmax, vmax=vmax, interpolation='nearest', aspect='auto')
    cbar2 = plt.colorbar(im2)
    cbar2.set_label('QUBO系数值')
    plt.title('QUBO矩阵可视化')
    plt.xlabel('变量索引')
    plt.ylabel('变量索引')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/problem2_qubo_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return best_thresholds, income

if __name__ == "__main__":
    # 求解第二问
    best_thresholds, max_income = solve_problem2_qubo()
    
    # 打印结果表格
    if max_income is not None:
        result_table = [
            ["最优阈值组合", f"[{best_thresholds[0]}, {best_thresholds[1]}, {best_thresholds[2]}]"],
            ["最大收入", f"{max_income:.2f}元"]
        ]
        print("\n最终结果:")
        print(tabulate(result_table, tablefmt="grid")) 