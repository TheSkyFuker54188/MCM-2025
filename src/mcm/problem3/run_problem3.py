from .solver3 import evaluate_pitch, export_result3
import os

def grid_search():
    print("开始第三问网格搜索最小螺距...")
    levels = [0.1, 0.01, 0.001]
    records = []
    # 方向: 如果 0.55 已进入则向下，不进入则先向下直到进入（题目期望最小螺距）
    p_start = 0.55
    print(f"步骤1: 评估初始螺距 {p_start}")
    entered, r_head, t_stop, collided, pair = evaluate_pitch(p_start)
    records.append(dict(p=p_start, entered=entered, r_head=r_head, t_stop=t_stop, collided=collided))
    if not entered:
        print("初始螺距无法进入，向下搜索可行点...")
        p = p_start - 0.1
        while p >= 0.05:
            e, r, ts, col, pr = evaluate_pitch(round(p,3))
            records.append(dict(p=round(p,3), entered=e, r_head=r, t_stop=ts, collided=col))
            if e:
                entered = True
                p_start = p
                break
            p -= 0.1
        if not entered:
            print("未找到可行螺距!")
            return None, records
    
    print(f"步骤2: 逐级网格搜索 (步长: {levels})")
    # 逐级减步长确定临界
    p_feasible = p_start  # 当前最大可行
    for level_idx, step in enumerate(levels):
        print(f"  级别{level_idx+1}: 步长={step}")
        p_try = p_feasible + step
        # 向上扩展直到不可行
        while True:
            e, r, ts, col, pr = evaluate_pitch(round(p_try,3))
            records.append(dict(p=round(p_try,3), entered=e, r_head=r, t_stop=ts, collided=col))
            if e:
                p_feasible = round(p_try,3)
                p_try += step
                if p_try > 2.5:  # 安全上界
                    break
            else:
                break
        # 下一层细化：从当前 p_feasible 向下扫描找到仍可行的最小 p (step 分辨率)
        scan_start = p_feasible
        p_scan = scan_start - step
        last_good = scan_start
        while p_scan >= 0.05 - 1e-12:
            e, r, ts, col, pr = evaluate_pitch(round(p_scan,3))
            records.append(dict(p=round(p_scan,3), entered=e, r_head=r, t_stop=ts, collided=col))
            if e:
                last_good = round(p_scan,3)
                p_scan -= step
            else:
                break
        p_feasible = last_good
        print(f"    当前可行上界: {p_feasible}")
    
    p_min = round(p_feasible,3)
    print(f"步骤3: 验证临界性 p_min={p_min}")
    # 验证
    e_plus, *_ = evaluate_pitch(p_min + 0.001)
    e_minus, *_ = evaluate_pitch(max(0.05, p_min - 0.001))
    records.append(dict(p=p_min+0.001, entered=e_plus, note='p_min+0.001'))
    records.append(dict(p=p_min-0.001, entered=e_minus, note='p_min-0.001'))
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
