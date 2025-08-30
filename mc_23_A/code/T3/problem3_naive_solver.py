import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tabulate import tabulate
from itertools import combinations
import time
import multiprocessing
from tqdm import tqdm
import matplotlib.font_manager as fm

# 设置中文显示
# 尝试更稳健地设置中文字体（优先 Windows 常见字体）
preferred_fonts = ['Microsoft YaHei', 'Microsoft YaHei UI', 'SimHei', 'SimSun', 'PingFang SC', 'Heiti SC']
available = {f.name for f in fm.fontManager.ttflist}
for pf in preferred_fonts:
    if pf in available:
        plt.rcParams['font.sans-serif'] = [pf]
        break
else:
    # 如果没有找到首选字体，保留原有设置为 SimHei（可能不可用）
    plt.rcParams['font.sans-serif'] = ['SimHei']

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

def solve_problem3(data_path=None, output_dir=None, use_multiprocessing=True, top_k=10):
    """
    求解第三问：从100张信用卡中找出最优的3张卡片组合及其阈值
    
    参数:
    data_path: 数据文件路径
    output_dir: 输出结果保存目录
    use_multiprocessing: 是否使用多进程计算
    top_k: 记录最优的前k个组合
    
    返回:
    best_cards: 最优卡片组合 [卡片1, 卡片2, 卡片3]
    best_thresholds: 最优阈值组合 [阈值1, 阈值2, 阈值3]
    max_income: 最大收入
    """
    start_time = time.time()
    print("开始求解第三问...")
    
    # 根据脚本位置确定项目根目录，支持相对或绝对路径输入
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    if data_path is None:
        data_path = os.path.join(project_root, 'data', 'data_100.csv')
    elif not os.path.isabs(data_path):
        # 相对路径相对于项目根
        data_path = os.path.join(project_root, data_path)

    if output_dir is None:
        output_dir = os.path.join(project_root, 'results', 'problem3')
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
    
    # 提取100张卡的通过率和坏账率数据
    t_rates = {}  # 通过率
    h_rates = {}  # 坏账率
    
    for card_id in range(1, 101):
        t_rates[card_id] = data[f't_{card_id}'].values
        h_rates[card_id] = data[f'h_{card_id}'].values
    
    print(f"开始计算所有可能的卡片组合和阈值组合...")
    
    # 生成所有不重复的三张卡组合
    card_combinations = list(combinations(range(1, 101), 3))
    print(f"总共有 {len(card_combinations)} 种不同的卡片组合")
    
    # 定义收入计算函数
    def calculate_income(card_combo, threshold_combo):
        a, b, c = card_combo
        i, j, k = threshold_combo
        
        t1 = t_rates[a][i-1]  # 注意：阈值从1开始，索引从0开始
        t2 = t_rates[b][j-1]
        t3 = t_rates[c][k-1]
        
        h1 = h_rates[a][i-1]
        h2 = h_rates[b][j-1]
        h3 = h_rates[c][k-1]
        
        # 计算总平均通过率和总平均坏账率
        T = (t1 + t2 + t3) / 3
        H = (h1 + h2 + h3) / 3
        
        # 计算最终收入
        income = M * T * (r - (1 + r) * H)
        
        return income, T, H
    
    # 定义处理一批卡片组合的函数
    def process_batch(card_batch):
        batch_results = []
        thresholds = range(1, 11)
        
        for card_combo in card_batch:
            for i in thresholds:
                for j in thresholds:
                    for k in thresholds:
                        income, T, H = calculate_income(card_combo, (i, j, k))
                        batch_results.append({
                            '卡片组合': card_combo,
                            '阈值组合': (i, j, k),
                            '总平均通过率': T,
                            '总平均坏账率': H,
                            '最终收入': income
                        })
        
        # 只返回这批计算中收入最高的top_k个结果
        sorted_results = sorted(batch_results, key=lambda x: x['最终收入'], reverse=True)
        return sorted_results[:top_k]
    
    # 计算结果
    all_top_results = []
    
    if use_multiprocessing:
        # 使用多进程加速计算
        num_cores = multiprocessing.cpu_count()
        print(f"使用 {num_cores} 个CPU核心进行并行计算...")
        
        # 将卡片组合分成多个批次
        batch_size = len(card_combinations) // num_cores
        if batch_size == 0:
            batch_size = 1
        
        batches = [card_combinations[i:i+batch_size] for i in range(0, len(card_combinations), batch_size)]
        
        with multiprocessing.Pool(processes=num_cores) as pool:
            # 使用tqdm显示进度
            for batch_results in tqdm(pool.imap_unordered(process_batch, batches), total=len(batches), desc="处理批次"):
                all_top_results.extend(batch_results)
    else:
        # 单进程计算
        print("使用单进程计算...")
        all_top_results = process_batch(card_combinations)
    
    # 找出全局最优组合
    all_top_results.sort(key=lambda x: x['最终收入'], reverse=True)
    top_results = all_top_results[:top_k]
    
    # 提取最优结果
    best_result = top_results[0]
    best_cards = best_result['卡片组合']
    best_thresholds = best_result['阈值组合']
    max_income = best_result['最终收入']
    
    # 输出最优结果
    print("\n最优结果:")
    print(f"最优卡片组合: {best_cards}")
    print(f"最优阈值组合: {best_thresholds}")
    print(f"最大收入: {max_income:.2f}元")
    
    # 将最优结果保存到文件
    with open(f"{output_dir}/problem3_best_naive_result.txt", 'w', encoding='utf-8-sig') as f:
        f.write(f"最优卡片组合: {best_cards}\n")
        f.write(f"最优阈值组合: {best_thresholds}\n")
        f.write(f"最大收入: {max_income:.2f}元\n")
        f.write(f"总平均通过率: {best_result['总平均通过率']:.4f}\n")
        f.write(f"总平均坏账率: {best_result['总平均坏账率']:.4f}\n")
        
        # 每张卡的通过率和坏账率
        f.write("\n每张卡的详细信息:\n")
        for i, card_id in enumerate(best_cards):
            threshold = best_thresholds[i]
            t_rate = t_rates[card_id][threshold-1]
            h_rate = h_rates[card_id][threshold-1]
            f.write(f"卡片 {card_id} (阈值 {threshold}): 通过率 = {t_rate:.4f}, 坏账率 = {h_rate:.4f}\n")
    
    # 保存前k个最优结果到CSV文件
    top_results_df = pd.DataFrame([
        {
            '排名': i+1,
            '卡片1': result['卡片组合'][0],
            '卡片2': result['卡片组合'][1],
            '卡片3': result['卡片组合'][2],
            '阈值1': result['阈值组合'][0],
            '阈值2': result['阈值组合'][1],
            '阈值3': result['阈值组合'][2],
            '总平均通过率': result['总平均通过率'],
            '总平均坏账率': result['总平均坏账率'],
            '最终收入': result['最终收入']
        }
        for i, result in enumerate(top_results)
    ])
    top_results_df.to_csv(f"{output_dir}/problem3_top_{top_k}_results.csv", index=False, encoding='utf-8-sig')
    print(f"已保存前{top_k}个最优结果到: {output_dir}/problem3_top_{top_k}_results.csv")
    
    # 绘制前10个最优组合的收入对比图
    plt.figure(figsize=(12, 8))
    top_n = min(10, len(top_results))
    incomes = [result['最终收入'] for result in top_results[:top_n]]
    labels = [f"组合{i+1}" for i in range(top_n)]
    
    # 计算相对于最低值的差值，更好地显示差异
    min_income = min(incomes)
    max_income = max(incomes)
    income_range = max_income - min_income
    
    # 如果差异很小，使用相对差值来显示
    if income_range < max_income * 0.01:  # 如果差异小于1%
        relative_incomes = [(income - min_income) for income in incomes]
        
        bars = plt.bar(labels, relative_incomes, color='steelblue')
        plt.xlabel('组合排名')
        plt.ylabel(f'相对于最低收入的增量 (元)\n最低收入: {min_income:.2f}元')
        plt.title(f'前{top_n}个最优组合的收入增量对比')
        
        # 添加数值标签 - 显示实际收入和增量
        for i, (bar, actual_income, relative_income) in enumerate(zip(bars, incomes, relative_incomes)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + income_range * 0.01,
                    f'{actual_income:.2f}\n(+{relative_income:.2f})',
                    ha='center', va='bottom', fontsize=9)
    else:
        # 正常显示
        bars = plt.bar(labels, incomes, color='steelblue')
        plt.xlabel('组合排名')
        plt.ylabel('最终收入 (元)')
        plt.title(f'前{top_n}个最优组合的收入对比')
        
        # 添加数值标签
        for i, (bar, income) in enumerate(zip(bars, incomes)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + income_range * 0.01,
                    f'{income:.2f}',
                    ha='center', va='bottom', fontsize=9)
    
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/problem3_top_combinations_comparison.png", dpi=300, bbox_inches='tight')
    print(f"已保存收入对比图到: {output_dir}/problem3_top_combinations_comparison.png")
    plt.close()
    
    # 计算运行时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n计算完成！运行时间: {elapsed_time:.2f} 秒")
    
    return best_cards, best_thresholds, max_income

def early_termination_solve(data_path=None, output_dir=None, batch_size=1000, max_batches=None):
    """
    使用批处理和提前终止策略的第三问求解器
    
    参数:
    data_path: 数据文件路径
    output_dir: 输出结果保存目录
    batch_size: 每批处理的卡片组合数量
    max_batches: 最大批次数量，如果为None则处理所有批次
    
    返回:
    best_cards: 最优卡片组合 [卡片1, 卡片2, 卡片3]
    best_thresholds: 最优阈值组合 [阈值1, 阈值2, 阈值3]
    max_income: 最大收入
    """
    start_time = time.time()
    print("开始求解第三问 (使用批处理和提前终止策略)...")
    
    # 根据脚本位置确定项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    if data_path is None:
        data_path = os.path.join(project_root, 'data', 'data_100.csv')
    elif not os.path.isabs(data_path):
        data_path = os.path.join(project_root, data_path)

    if output_dir is None:
        output_dir = os.path.join(project_root, 'results', 'problem3')
    elif not os.path.isabs(output_dir):
        output_dir = os.path.join(project_root, output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据
    try:
        data = pd.read_csv(data_path)
        print(f"成功读取数据文件: {data_path}")
    except Exception as e:
        print(f"读取数据文件失败: {e}")
        return None, None, None
    
    # 贷款资金和利率
    M = 1000000
    r = 0.08
    
    # 提取100张卡的通过率和坏账率数据
    t_rates = {}
    h_rates = {}
    
    for card_id in range(1, 101):
        t_rates[card_id] = data[f't_{card_id}'].values
        h_rates[card_id] = data[f'h_{card_id}'].values
    
    # 首先评估每张卡在最佳阈值下的收入情况
    card_best_incomes = {}
    card_best_thresholds = {}
    
    print("预评估每张卡片的最佳阈值和收入...")
    for card_id in range(1, 101):
        card_incomes = []
        for threshold in range(1, 11):
            t = t_rates[card_id][threshold-1]
            h = h_rates[card_id][threshold-1]
            income = M * t * (r - (1 + r) * h)
            card_incomes.append((income, threshold))
        
        best_income, best_threshold = max(card_incomes)
        card_best_incomes[card_id] = best_income
        card_best_thresholds[card_id] = best_threshold
    
    # 根据单卡最佳收入对卡片进行排序
    sorted_cards = sorted(range(1, 101), key=lambda x: card_best_incomes[x], reverse=True)
    
    # 生成卡片组合，优先考虑最佳收入较高的卡片组合
    print("生成卡片组合...")
    card_combinations = list(combinations(sorted_cards, 3))
    print(f"总共有 {len(card_combinations)} 种不同的卡片组合")
    
    # 定义收入计算函数
    def calculate_income(card_combo, threshold_combo):
        a, b, c = card_combo
        i, j, k = threshold_combo
        
        t1 = t_rates[a][i-1]
        t2 = t_rates[b][j-1]
        t3 = t_rates[c][k-1]
        
        h1 = h_rates[a][i-1]
        h2 = h_rates[b][j-1]
        h3 = h_rates[c][k-1]
        
        # 计算总平均通过率和总平均坏账率
        T = (t1 + t2 + t3) / 3
        H = (h1 + h2 + h3) / 3
        
        # 计算最终收入
        income = M * T * (r - (1 + r) * H)
        
        return income, T, H
    
    # 批处理计算
    best_result = None
    max_income = float('-inf')
    all_results = []  # 存储所有结果用于后续分析
    
    num_batches = (len(card_combinations) + batch_size - 1) // batch_size
    if max_batches is not None:
        num_batches = min(num_batches, max_batches)
    
    print(f"开始处理 {num_batches} 批卡片组合 (每批 {batch_size} 个组合)...")
    
    for batch_idx in tqdm(range(num_batches)):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(card_combinations))
        card_batch = card_combinations[batch_start:batch_end]
        
        for card_combo in card_batch:
            # 对每个卡片组合，尝试最佳阈值组合
            best_threshold_combo = (
                card_best_thresholds.get(card_combo[0], 1),
                card_best_thresholds.get(card_combo[1], 1),
                card_best_thresholds.get(card_combo[2], 1)
            )
            
            # 计算初始收入
            income, T, H = calculate_income(card_combo, best_threshold_combo)
            
            # 如果当前组合的收入比已知的最大收入低20%以上，则跳过对该组合的详细计算
            if income < max_income * 0.8:
                continue
            
            # 否则进行局部搜索，遍历附近的阈值组合
            for i in range(max(1, best_threshold_combo[0]-1), min(11, best_threshold_combo[0]+2)):
                for j in range(max(1, best_threshold_combo[1]-1), min(11, best_threshold_combo[1]+2)):
                    for k in range(max(1, best_threshold_combo[2]-1), min(11, best_threshold_combo[2]+2)):
                        local_income, local_T, local_H = calculate_income(card_combo, (i, j, k))
                        
                        if local_income > income:
                            income, T, H = local_income, local_T, local_H
                            best_threshold_combo = (i, j, k)
            
            # 更新全局最优结果
            if income > max_income:
                max_income = income
                best_result = {
                    '卡片组合': card_combo,
                    '阈值组合': best_threshold_combo,
                    '总平均通过率': T,
                    '总平均坏账率': H,
                    '最终收入': income
                }
                
                print(f"\n发现新的最优组合 (批次 {batch_idx+1}/{num_batches}):")
                print(f"卡片组合: {card_combo}")
                print(f"阈值组合: {best_threshold_combo}")
                print(f"最终收入: {income:.2f}元")
            
            # 保存所有有效结果用于后续分析
            all_results.append({
                '卡片组合': card_combo,
                '阈值组合': best_threshold_combo,
                '总平均通过率': T,
                '总平均坏账率': H,
                '最终收入': income
            })
    
    if best_result:
        best_cards = best_result['卡片组合']
        best_thresholds = best_result['阈值组合']
        max_income = best_result['最终收入']
        
        # 保存最优结果
        with open(f"{output_dir}/problem3_best_result.txt", 'w', encoding='utf-8-sig') as f:
            f.write(f"最优卡片组合: {best_cards}\n")
            f.write(f"最优阈值组合: {best_thresholds}\n")
            f.write(f"最大收入: {max_income:.2f}元\n")
            f.write(f"总平均通过率: {best_result['总平均通过率']:.4f}\n")
            f.write(f"总平均坏账率: {best_result['总平均坏账率']:.4f}\n")
            
            # 每张卡的通过率和坏账率
            f.write("\n每张卡的详细信息:\n")
            for i, card_id in enumerate(best_cards):
                threshold = best_thresholds[i]
                t_rate = t_rates[card_id][threshold-1]
                h_rate = h_rates[card_id][threshold-1]
                f.write(f"卡片 {card_id} (阈值 {threshold}): 通过率 = {t_rate:.4f}, 坏账率 = {h_rate:.4f}\n")
        
        print(f"已保存最优结果到: {output_dir}/problem3_best_result.txt")
        
        # 保存所有结果到CSV文件
        if all_results:
            # 按收入排序
            all_results.sort(key=lambda x: x['最终收入'], reverse=True)
            top_k = min(20, len(all_results))  # 保存前20个结果
            
            results_df = pd.DataFrame([
                {
                    '排名': i+1,
                    '卡片1': result['卡片组合'][0],
                    '卡片2': result['卡片组合'][1],
                    '卡片3': result['卡片组合'][2],
                    '阈值1': result['阈值组合'][0],
                    '阈值2': result['阈值组合'][1],
                    '阈值3': result['阈值组合'][2],
                    '总平均通过率': result['总平均通过率'],
                    '总平均坏账率': result['总平均坏账率'],
                    '最终收入': result['最终收入']
                }
                for i, result in enumerate(all_results[:top_k])
            ])
            results_df.to_csv(f"{output_dir}/problem3_top_{top_k}_results.csv", index=False, encoding='utf-8-sig')
            print(f"已保存前{top_k}个结果到: {output_dir}/problem3_top_{top_k}_results.csv")
            
            # 绘制前10个最优组合的收入对比图
            top_n = min(10, len(all_results))
            incomes = [result['最终收入'] for result in all_results[:top_n]]
            labels = [f"组合{i+1}" for i in range(top_n)]
            
            plt.figure(figsize=(12, 8))
            
            # 计算相对于最低值的差值，更好地显示差异
            min_income = min(incomes)
            max_income_val = max(incomes)
            income_range = max_income_val - min_income
            
            # 如果差异很小，使用相对差值来显示
            if income_range < max_income_val * 0.01:  # 如果差异小于1%
                relative_incomes = [(income - min_income) for income in incomes]
                
                bars = plt.bar(labels, relative_incomes, color='steelblue')
                plt.xlabel('组合排名')
                plt.ylabel(f'相对于最低收入的增量 (元)\n最低收入: {min_income:.2f}元')
                plt.title(f'前{top_n}个最优组合的收入增量对比')
                
                # 添加数值标签 - 显示实际收入和增量
                for i, (bar, actual_income, relative_income) in enumerate(zip(bars, incomes, relative_incomes)):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + income_range * 0.01,
                            f'{actual_income:.2f}\n(+{relative_income:.2f})',
                            ha='center', va='bottom', fontsize=9)
            else:
                # 正常显示
                bars = plt.bar(labels, incomes, color='steelblue')
                plt.xlabel('组合排名')
                plt.ylabel('最终收入 (元)')
                plt.title(f'前{top_n}个最优组合的收入对比')
                
                # 添加数值标签
                for i, (bar, income) in enumerate(zip(bars, incomes)):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + income_range * 0.01,
                            f'{income:.2f}',
                            ha='center', va='bottom', fontsize=9)
            
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/problem3_top_combinations_comparison.png", dpi=300, bbox_inches='tight')
            print(f"已保存收入对比图到: {output_dir}/problem3_top_combinations_comparison.png")
            plt.close()
        
        # 计算运行时间
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n计算完成！运行时间: {elapsed_time:.2f} 秒")
        
        return best_cards, best_thresholds, max_income
    else:
        print("未找到有效的最优组合")
        return None, None, None

if __name__ == "__main__":
    # 设置参数
    # 全暴力搜索计算量太大，使用提前终止策略求解
    print("注意: 第三问涉及大量计算，将使用批处理和提前终止策略进行求解")
    best_cards, best_thresholds, max_income = early_termination_solve(batch_size=5000, max_batches=100)
    
    # 打印结果表格
    if max_income is not None:
        result_table = [
            ["最优卡片组合", f"{best_cards}"],
            ["最优阈值组合", f"{best_thresholds}"],
            ["最大收入", f"{max_income:.2f}元"]
        ]
        print("\n最终结果:")
        print(tabulate(result_table, tablefmt="grid")) 