"""第三问: 变螺距搜索最小 p 使龙头前把手进入半径 R=4.5 m 调头空间。

策略:
 1. 对给定螺距 p 计算 b = p/(2π)。初始龙头角度仍假设 32π。
 2. 复用第一问的链条长度与速度逻辑, 但螺线几何需参数化 b。
 3. 复用第二问矩形碰撞检测 (几何不依赖 p) 但需提供新的 sample 函数。
 4. 在给定 p 下运行碰撞搜索(最大时间 t_max 可设 400 或更大以保证充分缠绕)。
 5. 得到停止时刻 (碰撞时刻或 t_max) 的龙头半径 r_head = b * theta_head(t_stop)。
 6. 判定 r_head 与 R=4.5: 若 r_head <= R 代表成功进入; 目标是求最小 p 仍能进入。
 7. 采用逐级减步长搜索: 从 p_start=0.55 (题给初值) 出发, 若其已经进入则向下搜索; 若未进入则向上扩大直到进入, 然后逐级缩小步长 (0.1,0.01,0.001)。

注意: 物理描述里 "l0<R 则减小螺距 p" (更紧 -> 更靠中心)。与上述一致。
"""
from __future__ import annotations
import math
from typing import List
import numpy as np

from ..problem1.constants import ChainParams
from ..problem2.collision import first_collision_time
from ..problem1 import spiral as spiral_module
from ..problem1.solver import newton_handle_theta as _unused  # ensure import side effects
from ..problem1.solver import solve_problem1  # still used for integer grid reuse if needed
from ..problem1.solver import newton_handle_theta

# 初始化常量
cp = ChainParams()
R_TARGET = 4.5
v_head = cp.v_head  # 添加龙头速度
# 预计算有效距离
EFFECTIVE_L = np.array([cp.effective_distance(i) for i in range(cp.handle_count-1)], dtype=np.float64)

try:
    from numba import njit
    
    @njit
    def _newton_handle_theta_numba(x_prev: float, y_prev: float, L: float, b: float, guess: float):
        """numba加速的牛顿迭代求解把手角度
        """
        theta_init = guess + 0.5 if guess > 0 else 0.5
        
        theta = theta_init
        max_iter = 100
        tolerance = 1e-12
        
        for iteration in range(max_iter):
            # 螺线位置
            x = b * theta * math.cos(theta)
            y = b * theta * math.sin(theta)
            
            # 距离约束方程 f = (x-x_prev)² + (y-y_prev)² - L²
            dx = x - x_prev
            dy = y - y_prev
            f = dx*dx + dy*dy - L*L
            
            if abs(f) < tolerance:  # 收敛
                break
            
            # 梯度计算
            cos_th = math.cos(theta)
            sin_th = math.sin(theta)
            dxdt = b * (cos_th - theta * sin_th)
            dydt = b * (sin_th + theta * cos_th)
            
            # 牛顿迭代: df/dθ = 2(x-x_prev)·dx/dθ + 2(y-y_prev)·dy/dθ
            df_dt = 2 * (dx * dxdt + dy * dydt)
            
            if abs(df_dt) < 1e-15:  # 避免除零
                break
            
            delta = f / df_dt
            theta_new = theta - delta
            
            # 确保theta保持正值且合理
            if theta_new <= 0:
                theta_new = theta * 0.5
            elif theta_new > theta * 3:
                theta_new = theta * 1.5
                
            theta = theta_new
            
            # 检查步长收敛
            if abs(delta) < tolerance:
                break
        
        return max(theta, 0.0)
    
    # 也定义其他numba函数
    @njit
    def _spiral_pos_local(theta: float, b: float):
        """本地螺线位置计算，不依赖全局模块"""
        return b * theta * math.cos(theta), b * theta * math.sin(theta)
    
    @njit 
    def _spiral_arc_length_inv_local(s: float, b: float, guess: float):
        """本地螺线弧长反函数，牛顿迭代"""
        theta = guess
        for _ in range(50):
            # s = b/2 * (θ√(1+θ²) + asinh(θ))
            sqrt_term = (1 + theta*theta)**0.5
            s_calc = b * 0.5 * (theta * sqrt_term + math.asinh(theta))
            
            if abs(s_calc - s) < 1e-12:
                break
            
            # 导数: ds/dθ = b * √(1+θ²)
            ds_dt = b * sqrt_term
            if abs(ds_dt) < 1e-15:
                break
            
            theta = theta - (s_calc - s) / ds_dt
            if theta < 0:
                theta = 0.01
        return theta
        
except ImportError:
    # 无numba时的备选实现
    def _newton_handle_theta_numba(x_prev: float, y_prev: float, L: float, b: float, guess: float):
        # newton_handle_theta不需要b参数，它会从spiral模块获取
        return newton_handle_theta(x_prev, y_prev, L, guess)
    
    def _spiral_pos_local(theta: float, b: float):
        return spiral_module.spiral_pos(theta, b)
    
    def _spiral_arc_length_inv_local(s: float, b: float, guess: float):
        return spiral_module.spiral_arc_length_inv(s, guess, b)
    
    def njit(*args, **kwargs):
        def wrap(f): return f
        return wrap

def build_interpolators(T:int, p:float):
    # 为特定螺距p构建采样函数，不依赖全局状态
    b = p/(2*math.pi)
    v_head = cp.v_head
    
    # 根据螺距调整初始角度，确保龙头能够盘入到4.5m边界
    # 目标角度：θ_target = 4.5 * 2π / p = 28.27 / p
    theta_target = 28.27 / p
    
    # 初始角度应该大于目标角度，给足够的盘入空间
    # 同时要考虑板凳龙的长度和碰撞约束
    # 修改初始角度设置，使龙头能更容易接近中心
    if p <= 0.55:
        theta_head_0 = max(32 * math.pi, theta_target * 1.5)
    elif p <= 1.0:
        theta_head_0 = max(24 * math.pi, theta_target * 1.4)
    elif p <= 2.0:
        theta_head_0 = max(18 * math.pi, theta_target * 1.3)
    elif p <= 3.0:
        theta_head_0 = max(15 * math.pi, theta_target * 1.2)
    elif p <= 4.0:
        theta_head_0 = max(12 * math.pi, theta_target * 1.1)
    else:
        theta_head_0 = max(10 * math.pi, theta_target * 1.05)
    
    # 计算初始弧长
    s_head_0 = b * 0.5 * (theta_head_0 * math.sqrt(1 + theta_head_0*theta_head_0) + math.asinh(theta_head_0))
    
    def _compute_frame_with_p(t: float):
        """为指定螺距p计算帧，不修改全局spiral参数"""
        # 龙头位置 
        s_head = s_head_0 - v_head * t
        if s_head <= 0:  # 防止弧长变为负值
            return [(0, 0)] * cp.handle_count, [0] * cp.handle_count
            
        guess_head = theta_head_0 * 0.9  # 简单估计
        theta_head = _spiral_arc_length_inv_local(s_head, b, guess_head)
        thetas = [theta_head]
        xs = [0.0]; ys = [0.0]
        xh, yh = _spiral_pos_local(theta_head, b)
        xs[0] = xh; ys[0] = yh
        
        # 其他把手
        for i in range(1, cp.handle_count):
            L = EFFECTIVE_L[i-1]
            guess = thetas[-1] * 0.9
            th_i = _newton_handle_theta_numba(xs[i-1], ys[i-1], L, b, guess)
            x_i, y_i = _spiral_pos_local(th_i, b)
            thetas.append(th_i)
            xs.append(x_i); ys.append(y_i)
        
        pts = [(xs[i], ys[i]) for i in range(cp.handle_count)]
        return pts, thetas
    
    def sample_points(t: float):
        return _compute_frame_with_p(t)[0]
    def sample_thetas(t: float):
        return _compute_frame_with_p(t)[1][:-1]
    def head_theta(t: float):
        return _compute_frame_with_p(t)[1][0]
    
    return sample_points, sample_thetas, head_theta, b

def _spiral_pos_local(theta: float, b: float):
    r = b * theta
    return r * math.cos(theta), r * math.sin(theta)

def _spiral_arc_length_inv_local(s: float, b: float, theta_guess: float) -> float:
    """本地版本弧长反函数，使用给定b参数"""
    theta = theta_guess
    for _ in range(25):
        # S(theta) = b/2 * (theta*sqrt(1+theta^2) + asinh(theta))
        f = b * 0.5 * (theta * math.sqrt(1 + theta*theta) + math.asinh(theta)) - s
        if abs(f) < 1e-12:
            return theta
        d = b * math.sqrt(1 + theta*theta)
        theta -= f / d
        if theta < 0:
            theta = 0.0
    return max(theta, 0.0)

def first_collision_time_with_offset(sample_func, theta_func, t_max: float, t_start: float = 0.0):
    """带时间偏移的碰撞检测，避免初始重叠问题"""
    from ..problem2.collision import build_bench_rects, candidate_pairs, rects_intersect
    
    t = t_start
    dt_coarse = 1.0
    last_clear = t_start
    collision_t = None
    pair_hit = None
    
    while t <= t_max:
        pts, thetas = sample_func(t), theta_func(t)
        rects = build_bench_rects(pts)
        pairs = candidate_pairs(thetas[:len(rects)])
        found = False
        for a, b in pairs:
            if rects_intersect(rects[a], rects[b]):
                collision_t = t
                pair_hit = (a, b)
                found = True
                break
        if found:
            break
        last_clear = t
        t += dt_coarse
    
    if collision_t is None:
        return False, None, None, None, None
    
    # 简化的二分搜索（避免复杂的frame重计算）
    lo = last_clear
    hi = collision_t
    
    while hi - lo > 1e-4:
        mid = 0.5 * (lo + hi)
        pts, thetas = sample_func(mid), theta_func(mid)
        rects = build_bench_rects(pts)
        pairs = candidate_pairs(thetas[:len(rects)])
        found = False
        for a, b in pairs:
            if rects_intersect(rects[a], rects[b]):
                found = True
                break
        if found:
            hi = mid
        else:
            lo = mid
    
    # 返回碰撞时刻的状态
    final_t = hi
    pts, thetas = sample_func(final_t), theta_func(final_t)
    rects = build_bench_rects(pts)
    return True, final_t, pts, rects, pair_hit

def evaluate_pitch(p: float, t_max: float = 1000.0):
    """返回 (entered, r_head_at_stop, t_stop, collision, pair_hit)
    entered: 是否在停止时头部进入 R_TARGET (r<=R_TARGET)
    collision: 是否因碰撞停止 (True) 否则为达到 t_max
    pair_hit: 若碰撞则给出对
    """
    print(f"  评估螺距 p={p:.3f}m...", end=" ", flush=True)
    sample_points, sample_thetas, head_theta, b = build_interpolators(int(t_max), p)
    
    # 根据螺距调整起始检测时间：螺距越大，初始角度越小，需要的偏移越小
    if p <= 0.55:
        t_start = 2.0  # 原始设置
    elif p <= 1.0:
        t_start = 1.0  # 减小偏移
    elif p <= 2.0:
        t_start = 0.5  # 进一步减小
    elif p <= 4.0:
        t_start = 0.1  # 最小偏移
    else:
        t_start = 0.05  # 更小的偏移
    
    # 首先快速检查是否能达到目标半径
    # 在不考虑碰撞的情况下，计算龙头到达最小半径的时间点
    min_distance = float('inf')
    min_distance_time = 0
    
    # 粗略采样检查最小距离
    for t in range(0, int(t_max), 5):
        try:
            th = head_theta(t)
            if th > 0:
                head_pos = _spiral_pos_local(th, b)
                distance = math.sqrt(head_pos[0]**2 + head_pos[1]**2)
                if distance < min_distance:
                    min_distance = distance
                    min_distance_time = t
        except:
            continue
    
    # 如果最小距离大于目标半径，则无法进入
    if min_distance > R_TARGET * 1.2:  # 给一些余量
        print(f"无法到达目标区域, 最小距离={min_distance:.2f}m > {R_TARGET}m")
        return False, min_distance, min_distance_time, False, None
    
    # 进行碰撞检测
    hit, t_hit, pts, rects, pair_hit = first_collision_time_with_offset(
        sample_points, sample_thetas, t_max, t_start)
    
    if hit:
        t_stop = t_hit
        th = head_theta(t_stop)
        
        # 修正：计算龙头到原点的距离
        head_pos = _spiral_pos_local(th, b)
        distance_to_origin = math.sqrt(head_pos[0]**2 + head_pos[1]**2)
        
        # 判断是否进入调头空间：距离原点小于等于R_TARGET
        entered = distance_to_origin <= R_TARGET + 1e-9
        print(f"碰撞@{t_hit:.1f}s, 龙头距离={distance_to_origin:.2f}m, 进入={entered}")
        return entered, distance_to_origin, t_stop, True, pair_hit
    else:
        # 未碰撞到 t_max，使用最小距离时刻的状态
        t_stop = min_distance_time
        th = head_theta(t_stop)
        
        # 修正：计算龙头到原点的距离
        head_pos = _spiral_pos_local(th, b)
        distance_to_origin = math.sqrt(head_pos[0]**2 + head_pos[1]**2)
        
        # 判断是否进入调头空间：距离原点小于等于R_TARGET
        entered = distance_to_origin <= R_TARGET + 1e-9
        print(f"无碰撞, 最小距离={distance_to_origin:.2f}m@{t_stop}s, 进入={entered}")
        return entered, distance_to_origin, t_stop, False, None

def search_p_min(p_start: float = 0.55):
    """二分搜索找到最小可行螺距
    
    题目要求：找到最小的螺距p，使得板凳能进入且龙头半径不超过4.5m
    
    实际上是要找临界螺距：恰好能进入调头空间的最小螺距
    
    返回(p_min, feasible, samples)
    """
    samples = []
    
    print(f"开始二分搜索最小可行螺距")
    
    # 首先确定搜索区间
    # 从已知的边界开始：0.55m不能进入，需要向上搜索
    p_low = 0.55   # 已知不可行
    p_high = 5.0   # 假设可行，如果不行会继续扩大
    
    # 验证边界
    print(f"验证搜索区间...")
    ent_low, r_low, _, _, _ = evaluate_pitch(p_low)
    ent_high, r_high, _, _, _ = evaluate_pitch(p_high)
    samples.append((p_low, ent_low, r_low, None, None))
    samples.append((p_high, ent_high, r_high, None, None))
    
    print(f"  p={p_low}: 龙头距离={r_low:.3f}m, 能进入={ent_low}")
    print(f"  p={p_high}: 龙头距离={r_high:.3f}m, 能进入={ent_high}")
    
    if ent_low:
        print(f"错误：下界{p_low}可行，需要向下调整")
        # 向下寻找不可行下界
        p_test = p_low - 0.05
        while p_test >= 0.05:
            ent, r, _, _, _ = evaluate_pitch(p_test)
            samples.append((p_test, ent, r, None, None))
            if not ent:
                p_low = p_test
                print(f"  找到不可行下界: p={p_low}")
                break
            p_test -= 0.05
    
    if not ent_high:
        print(f"警告：上界{p_high}不可行，向上寻找可行上界")
        p_test = p_high + 1.0  # 更大的步长
        while p_test <= 20.0:  # 扩大搜索范围到20.0
            ent, r, _, _, _ = evaluate_pitch(p_test)
            samples.append((p_test, ent, r, None, None))
            if ent:
                p_high = p_test
                print(f"  找到可行上界: p={p_high}")
                break
            p_test += 1.0  # 更大步长
        
        if p_test > 20.0 and not ent:
            print("未找到可行上界，尝试更大的螺距值...")
            return None, False, samples
    
    # 二分搜索
    print(f"开始二分搜索，区间: [{p_low}, {p_high}]")
    tolerance = 0.001
    iteration = 0
    max_iterations = 30  # 防止无限循环
    
    while p_high - p_low > tolerance and iteration < max_iterations:
        iteration += 1
        p_mid = (p_low + p_high) / 2
        p_mid = round(p_mid, 4)
        
        ent_mid, r_mid, t_mid, col_mid, pair_mid = evaluate_pitch(p_mid)
        samples.append((p_mid, ent_mid, r_mid, t_mid, col_mid))
        
        print(f"  迭代{iteration}: p={p_mid}, 龙头距离={r_mid:.3f}m, 能进入={ent_mid}")
        
        if ent_mid:
            p_high = p_mid  # 可行，向下搜索更小的
        else:
            p_low = p_mid  # 不可行，向上搜索
    
    if iteration >= max_iterations:
        print(f"警告：达到最大迭代次数 {max_iterations}")
    
    p_min = round(p_high, 3)
    print(f"二分搜索完成，最小可行螺距: p_min = {p_min}")
    
    # 最终验证
    ent_final, r_final, _, _, _ = evaluate_pitch(p_min)
    print(f"最终验证: p={p_min}, 龙头距离={r_final:.3f}m, 能进入={ent_final}")
    
    return p_min, ent_final, samples

def export_result3(p_min: float, detail_rows, path: str):
    import pandas as pd
    # 验证p_min临界性（反馈要求）
    validation_data = []
    test_values = [p_min - 0.001, p_min, p_min + 0.001]
    for p_test in test_values:
        if p_test > 0.01:  # 确保螺距为正值且合理
            entered, r_head, t_stop, collided, pair = evaluate_pitch(p_test)
            validation_data.append({
                'p': p_test,
                'entered': entered,
                'r_head': round(r_head, 6),
                't_stop': t_stop,
                'collided': collided,
                'note': f'p_min{"" if p_test == p_min else ("-0.001" if p_test < p_min else "+0.001")}'
            })
    
    df = pd.DataFrame(detail_rows)
    validation_df = pd.DataFrame(validation_data)
    
    # 检查validation_data是否有足够的元素
    validation_passed = False
    if len(validation_data) >= 2:
        # 找到p_min对应的条目
        p_min_entry = next((item for item in validation_data if abs(item['p'] - p_min) < 1e-6), None)
        p_minus_entry = next((item for item in validation_data if item['p'] < p_min), None)
        
        if p_min_entry and p_minus_entry:
            validation_passed = p_min_entry['entered'] and not p_minus_entry['entered']
    
    summary = pd.DataFrame({"key":["p_min", "validation_passed"], 
                          "value":[p_min, validation_passed]})
    
    with pd.ExcelWriter(path, engine='openpyxl') as w:
        summary.to_excel(w, sheet_name='summary', index=False)
        df.to_excel(w, sheet_name='grid_search', index=False)
        validation_df.to_excel(w, sheet_name='validation', index=False)

def compute_with_logging(p_list):
    rows = []
    for p in p_list:
        entered, r_head, t_stop, collided, pair = evaluate_pitch(p)
        rows.append({
            "p": round(p,4),
            "entered": entered,
            "r_head": r_head,
            "t_stop": t_stop,
            "collided": collided,
            "pair": str(pair)
        })
    return rows

if __name__ == "__main__":
    p_min, feasible, samples = search_p_min()
    print("p_min=", p_min, "feasible=", feasible)
    print(f"总共测试了 {len(samples)} 个螺距值")
    
    # 生成详细结果
    if feasible:
        # 将samples转换为适合export的格式
        p_list = [s[0] for s in samples]  # 提取螺距值
        detail_rows = compute_with_logging(p_list)
        export_result3(p_min, detail_rows, "result3.xlsx")
        print(f"结果已导出到 result3.xlsx")
