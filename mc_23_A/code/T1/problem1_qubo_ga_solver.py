import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tabulate import tabulate
import random
from deap import base, creator, tools, algorithms

def solve_problem1_qubo_ga(data_path=None, output_dir=None):
    """
    使用QUBO模型+遗传算法求解第一问：找出100张信用卡中最优的单张卡片及其阈值设置
    
    参数:
    data_path: 数据文件路径
    output_dir: 输出结果保存目录
    
    返回:
    max_income_card: 最优信用卡编号
    max_income_threshold: 最优阈值编号
    max_income: 最大收入
    """
    print("开始使用QUBO模型+遗传算法求解第一问...")
    
    # 根据脚本位置确定项目根目录，支持相对或绝对路径输入
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    if data_path is None:
        data_path = os.path.join(project_root, 'data', 'data_100.csv')
    elif not os.path.isabs(data_path):
        # 相对路径相对于项目根
        data_path = os.path.join(project_root, data_path)

    if output_dir is None:
        output_dir = os.path.join(project_root, 'results', 'problem1_qubo_ga')
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
    
    # 存储卡号和阈值的映射
    var_mapping = []
    
    # 记录每个决策变量对应的收入
    incomes = []
    
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
            
            # 记录变量映射和收入
            var_mapping.append((card_id, threshold_idx + 1))
            incomes.append(income)
    
    # 问题规模
    n_vars = len(var_mapping)  # 1000个变量
    
    #! 罚函数系数
    A = 1e4  # 足够大的正数
    
    # 定义遗传算法适应度函数
    def evaluate(individual):
        # 计算目标函数值（最小化负收入）
        income_term = 0
        for i in range(n_vars):
            if individual[i] == 1:
                income_term -= incomes[i]  # 负号是为了将最大化问题转换为最小化问题
        
        # 计算约束条件惩罚项
        sum_x = sum(individual)
        constraint_penalty = A * ((sum_x - 1) ** 2)
        
        return (income_term + constraint_penalty,)  # 注意返回值必须是元组
    
    # 设置遗传算法
    # 由于我们要最小化目标函数，所以适应度是要被最小化的
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # 最小化单目标
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    
    # 属性生成器 - 生成0或1
    toolbox.register("attr_bool", random.randint, 0, 1)
    
    # 结构初始化
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_vars)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # 遗传操作
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)  # 两点交叉
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)  # 位翻转变异，每个位的变异概率为5%
    toolbox.register("select", tools.selTournament, tournsize=3)  # 锦标赛选择
    
    # 创建初始种群
    pop_size = 100
    population = toolbox.population(n=pop_size)
    
    # 遗传算法参数
    CXPB = 0.5  # 交叉概率
    MUTPB = 0.2  # 变异概率
    NGEN = 200   #! 代数
    
    print(f"开始遗传算法求解，种群大小: {pop_size}，最大代数: {NGEN}...")
    
    # 记录每代的最佳适应度
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    
    # 运行遗传算法
    pop, log = algorithms.eaSimple(population, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, 
                                  stats=stats, verbose=True)
    
    # 获取最优个体
    best_ind = tools.selBest(pop, 1)[0]
    
    # 找出值为1的变量索引
    selected_idx = None
    for i in range(n_vars):
        if best_ind[i] == 1:
            selected_idx = i
            break
    
    if selected_idx is not None:
        # 获取对应的卡号和阈值
        max_income_card, max_income_threshold = var_mapping[selected_idx]
        max_income = incomes[selected_idx]
    else:
        # 如果遗传算法没有找到有效解，尝试强制选择一个解
        print("遗传算法未找到严格符合约束的解，选择最接近的解...")
        # 将所有变量设为0，然后选择收入最高的变量设为1
        best_idx = np.argmax(incomes)
        max_income_card, max_income_threshold = var_mapping[best_idx]
        max_income = incomes[best_idx]
    
    # 为验证遗传算法结果，也计算所有可能的收入并找出最大值
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
    
    # 找出最大收入（用于验证遗传算法结果）
    max_income_row = results_df.loc[results_df['最终收入'].idxmax()]
    brute_force_card = int(max_income_row['信用卡编号'])
    brute_force_threshold = int(max_income_row['阈值编号'])
    brute_force_income = max_income_row['最终收入']
    
    # 输出最优结果
    print("\nQUBO+遗传算法求解结果:")
    print(f"最优信用卡编号: {max_income_card}")
    print(f"最优阈值编号: {max_income_threshold}")
    print(f"最大收入: {max_income:.2f}元")
    
    print("\n暴力枚举验证结果:")
    print(f"最优信用卡编号: {brute_force_card}")
    print(f"最优阈值编号: {brute_force_threshold}")
    print(f"最大收入: {brute_force_income:.2f}元")
    
    # 验证两种方法结果是否一致
    if max_income_card == brute_force_card and max_income_threshold == brute_force_threshold:
        print("\n✓ QUBO+遗传算法结果与暴力枚举结果一致")
    else:
        print("\n✗ QUBO+遗传算法结果与暴力枚举结果不一致，请检查模型")
    
    # 将最优结果保存到文件
    with open(f"{output_dir}/problem1_best_result.txt", 'w', encoding='utf-8') as f:
        f.write(f"QUBO+遗传算法求解结果:\n")
        f.write(f"最优信用卡编号: {max_income_card}\n")
        f.write(f"最优阈值编号: {max_income_threshold}\n")
        f.write(f"最大收入: {max_income:.2f}元\n")
        f.write(f"对应通过率: {data[f't_{max_income_card}'].values[max_income_threshold - 1]:.4f}\n")
        f.write(f"对应坏账率: {data[f'h_{max_income_card}'].values[max_income_threshold - 1]:.4f}\n\n")
        
        f.write(f"暴力枚举验证结果:\n")
        f.write(f"最优信用卡编号: {brute_force_card}\n")
        f.write(f"最优阈值编号: {brute_force_threshold}\n")
        f.write(f"最大收入: {brute_force_income:.2f}元\n")
    
    # 保存遗传算法优化过程
    generations = range(len(log))
    min_fitness = [log[i]['min'] for i in range(len(log))]
    avg_fitness = [log[i]['avg'] for i in range(len(log))]
    
    plt.figure(figsize=(10, 6))
    plt.plot(generations, min_fitness, 'b-', label='最佳适应度')
    plt.plot(generations, avg_fitness, 'r-', label='平均适应度')
    plt.xlabel('代数')
    plt.ylabel('适应度值 (越小越好)')
    plt.title('遗传算法优化过程')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/genetic_algorithm_optimization.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制最优卡片的收入图表
    optimal_card_results = results_df[results_df['信用卡编号'] == max_income_card]
    plt.figure(figsize=(10, 6))
    plt.plot(optimal_card_results['阈值编号'], optimal_card_results['最终收入'], 'bo-', linewidth=2)
    plt.axvline(x=max_income_threshold, color='r', linestyle='--', label=f'最优阈值 ({max_income_threshold})')
    plt.xlabel('阈值编号')
    plt.ylabel('最终收入 (元)')
    plt.title(f'信用卡 {max_income_card} 在不同阈值下的收入 (QUBO+遗传算法求解)')
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
    
    # 避免重复创建creator对象
    if 'FitnessMin' not in creator.__dict__:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
    
    # 求解第一问（使用基于脚本位置的默认路径）
    max_income_card, max_income_threshold, max_income = solve_problem1_qubo_ga()
    
    # 打印结果表格
    if max_income is not None:
        result_table = [
            ["最优信用卡编号", max_income_card],
            ["最优阈值编号", max_income_threshold],
            ["最大收入", f"{max_income:.2f}元"]
        ]
        print("\n最终结果:")
        print(tabulate(result_table, tablefmt="grid")) 