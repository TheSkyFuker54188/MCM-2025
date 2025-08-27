from .solver3 import evaluate_pitch, export_result3
import os

def grid_search():
    print("开始第三问网格搜索最小螺距...")
    levels = [0.1, 0.01, 0.001]
    records = []
    
    p_start = 0.55
    print(f"步骤1: 评估初始螺距 {p_start}")
    entered, r_head, t_stop, collided, pair = evaluate_pitch(p_start)
    records.append(dict(p=p_start, entered=entered, r_head=r_head, t_stop=t_stop, collided=collided))
    
    if not entered:
        print("初始螺距无法进入，向上搜索找到第一个可行点...")
        # 使用更大范围的搜索
        search_points = [
            # 小步长搜索较小的螺距
            *[round(0.55 + i*0.05, 2) for i in range(1, 30)],  # 0.6 to 2.0
            # 中等步长搜索中等螺距
            *[round(2.0 + i*0.2, 2) for i in range(1, 16)],    # 2.2 to 5.0
            # 大步长搜索较大的螺距
            *[round(5.0 + i*0.5, 2) for i in range(1, 21)],    # 5.5 to 15.0
            # 更大步长搜索非常大的螺距
            *[round(15.0 + i*1.0, 2) for i in range(1, 11)]    # 16.0 to 25.0
        ]
        
        found_feasible = False
        for p in search_points:
            e, r, ts, col, pr = evaluate_pitch(p)
            records.append(dict(p=p, entered=e, r_head=r, t_stop=ts, collided=col))
            
            if e:
                entered = True
                p_start = p
                print(f"找到第一个可行螺距: {p_start}m")
                found_feasible = True
                break
        
        if not found_feasible:
            print("未找到可行螺距!")
            return None, records
    
    print(f"步骤2: 逐级网格搜索 (步长: {levels})")
    # 现在需要找到最小的可行螺距（在能进入的前提下螺距越小越好）
    p_feasible = p_start  # 当前已知的可行螺距
    
    # 向下搜索找到不可行的下界
    print(f"  寻找不可行下界...")
    p_lower = None
    step = 0.05
    p_try = p_feasible - step
    
    while p_try >= 0.05:
        e, r, ts, col, pr = evaluate_pitch(p_try)
        records.append(dict(p=p_try, entered=e, r_head=r, t_stop=ts, collided=col))
        
        if not e:
            p_lower = p_try
            print(f"  找到不可行下界: {p_lower}m")
            break
        else:
            p_feasible = p_try  # 更新为更小的可行值
            p_try -= step
    
    if p_lower is None:
        print(f"  未找到不可行下界，当前最小可行螺距: {p_feasible}m")
        p_lower = max(0.05, p_feasible - 0.1)  # 设置一个人为的下界
    
    # 二分搜索找到精确的临界点
    p_upper = p_feasible
    
    for level_idx, step in enumerate(levels):
        print(f"  级别{level_idx+1}: 步长={step}")
        
        # 二分搜索找到临界点
        while p_upper - p_lower > step:
            p_mid = (p_upper + p_lower) / 2
            p_mid = round(p_mid, 6)
            
            e, r, ts, col, pr = evaluate_pitch(p_mid)
            records.append(dict(p=p_mid, entered=e, r_head=r, t_stop=ts, collided=col))
            
            if e:
                p_upper = p_mid  # 可行，向下搜索
            else:
                p_lower = p_mid  # 不可行，向上搜索
        
        print(f"    当前区间: [{p_lower}, {p_upper}]")
    
    p_min = round(p_upper, 3)  # 取可行的最小值
    print(f"步骤3: 验证临界性 p_min={p_min}")
    
    # 验证：p_min可行，p_min-0.001不可行
    e_curr, r_curr, *_ = evaluate_pitch(p_min)
    e_minus, r_minus, *_ = evaluate_pitch(p_min - 0.001)
    
    records.append(dict(p=p_min, entered=e_curr, r_head=r_curr, note='p_min'))
    records.append(dict(p=p_min-0.001, entered=e_minus, r_head=r_minus, note='p_min-0.001'))
    
    print(f"  p_min={p_min}: 进入={e_curr}, 距离={r_curr:.3f}m")
    print(f"  p_min-0.001={p_min-0.001}: 进入={e_minus}, 距离={r_minus:.3f}m")
    
    validation_passed = e_curr and not e_minus
    print(f"  验证结果: {'通过' if validation_passed else '失败'}")
    
    return p_min, records

def main():
    p_min, rows = grid_search()
    if p_min is None:
        print('未找到可行螺距区间')
        return
    out_path = os.path.abspath('result3.xlsx')
    export_result3(p_min, rows, out_path)
    print(f'最小螺距 p_min = {p_min:.3f} m, 已导出 result3.xlsx')

if __name__ == '__main__':
    main()
