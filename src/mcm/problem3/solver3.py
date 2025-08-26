"""第三问: 变螺距搜索最小 p 使龙头前把手进入半径 R=4.5 m 调头空间。

策略:
 1. 对给定螺距 p 计算 b = p/(2π)。初始龙头角度仍假设 32π。
 2. 复用第一问的链条长度与速度逻辑, 但螺线几何需参数化 b。
 3. 复用第二问矩形碰撞检测 (几何不依赖 p) 但需提供新的 sample 函数。
 4. 在给定 p 下运行碰撞搜索(最大时间 t_max 可设 400 或更大以保证充分缠绕)。
 5. 得到停止时刻 (碰撞时刻或 t_max) 的龙头半径 r_head = b * theta_head(t_stop)。
 6. 判定 r_head 与 R=4.5: 若 r_head <= R 代表成功进入; 目标是求最小 p 仍能进入。
 7. 采用逐级减步长搜索: 从 p_start=0.55 (题给初值) 出发, 若其已经进入则向下搜索; 若未进入则向上扩大直到进入, 然后逐级缩小步长 (0.1,0.01,0.001)。

注意: 物理描述里 “l0<R 则减小螺距 p” (更紧 -> 更靠中心)。与上述一致。
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
theta_head_0 = 32 * math.pi
v_head = cp.v_head  # 添加龙头速度
s_head_0 = spiral_module.spiral_arc_length(theta_head_0)  # 添加初始弧长
# 预计算有效距离
EFFECTIVE_L = np.array([cp.effective_distance(i) for i in range(cp.handle_count-1)], dtype=np.float64)

try:
    from numba import njit
    
    @njit
    def _newton_handle_theta_numba(x_prev: float, y_prev: float, L: float, b: float, guess: float):
        """numba加速的牛顿迭代求解把手角度，几何初始化策略"""
        # 几何初始化: 先根据前一个位置和距离约束进行合理估计
        r_prev = (x_prev*x_prev + y_prev*y_prev)**0.5
        if r_prev > 0:
            # 估计新半径应该在 r_prev ± L 范围内
            r_target = max(0.1, r_prev - L * 0.8)  # 偏向内侧
            theta_init = r_target / b if b > 1e-10 else guess
        else:
            theta_init = guess
        
        theta = theta_init
        for _ in range(50):  # 最大迭代次数
            # 螺线位置
            x = b * theta * math.cos(theta)
            y = b * theta * math.sin(theta)
            
            # 距离约束方程 f = (x-x_prev)² + (y-y_prev)² - L²
            dx = x - x_prev
            dy = y - y_prev
            f = dx*dx + dy*dy - L*L
            
            if abs(f) < 1e-12:  # 收敛
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
            theta = theta - delta
            
            # 防止theta过度偏离合理范围
            if theta < 0:
                theta = 0.1
            elif theta > 100:
                theta = 100
        
        return theta
    
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
        return newton_handle_theta(x_prev, y_prev, L, guess, b)
    
    def _spiral_pos_local(theta: float, b: float):
        return spiral_module.spiral_pos(theta, b)
    
    def _spiral_arc_length_inv_local(s: float, b: float, guess: float):
        return spiral_module.spiral_arc_length_inv(s, guess, b)
    
    def njit(*args, **kwargs):
        def wrap(f): return f
        return wrap

@njit(cache=True, fastmath=True)
def _newton_handle_theta_numba(x_prev: float, y_prev: float, L: float, b: float, theta_guess: float) -> float:
    """Numba版本的牛顿迭代求解把手角度"""
    if b <= 0:
        return 0.01
    
    # 初值基于几何估算
    dist_to_origin = math.sqrt(x_prev*x_prev + y_prev*y_prev)
    estimated_r = max(dist_to_origin - L, 0.1)
    th_init = max(estimated_r / b, 0.1)
    
    if theta_guess > 0.1:
        th_init = min(theta_guess * 0.9, th_init)
    
    th = th_init
    best_th = th
    best_error = 1e10
    
    for iteration in range(60):
        if th <= 0:
            th = 0.01
            
        r = b * th
        c = math.cos(th); s = math.sin(th)
        X = r * c; Y = r * s
        dx = X - x_prev; dy = Y - y_prev
        f = dx*dx + dy*dy - L*L
        
        error = abs(f)
        if error < best_error:
            best_error = error
            best_th = th
            
        if error < 1e-8:
            return th
            
        # 牛顿步长
        dX = b * c - r * s
        dY = b * s + r * c
        df = 2*dx*dX + 2*dy*dY
        
        if abs(df) < 1e-15:
            break
            
        step = f / df
        th_new = th - step
        
        # 约束到合理范围
        if th_new <= 0:
            th_new = th * 0.5
        elif th_new > th * 2:
            th_new = th * 1.1
            
        th = th_new
        
        # 防止振荡
        if iteration > 30 and error > 1e-4:
            th = best_th
            break
    
    return max(best_th, 0.01)

def build_interpolators(T:int, p:float):
    # 为特定螺距p构建采样函数，不依赖全局状态
    b = p/(2*math.pi)
    v_head = cp.v_head
    s_head_0 = b * 0.5 * (theta_head_0 * math.sqrt(1 + theta_head_0*theta_head_0) + math.asinh(theta_head_0))
    
    def _compute_frame_with_p(t: float):
        """为指定螺距p计算帧，不修改全局spiral参数"""
        # 龙头位置 
        s_head = s_head_0 - v_head * t
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

def evaluate_pitch(p: float, t_max: float = 800.0):
    """返回 (entered, r_head_at_stop, t_stop, collision, pair_hit)
    entered: 是否在停止时头部进入 R_TARGET (r<=R_TARGET)
    collision: 是否因碰撞停止 (True) 否则为达到 t_max
    pair_hit: 若碰撞则给出对
    """
    print(f"  评估螺距 p={p:.3f}m...", end=" ", flush=True)
    sample_points, sample_thetas, head_theta, b = build_interpolators(int(t_max), p)
    
    # 根据螺距调整起始检测时间：螺距越小，需要的偏移越大
    if p < 0.15:
        t_start = 10.0
    elif p < 0.3:
        t_start = 5.0
    elif p < 0.5:
        t_start = 2.0
    else:
        t_start = 1.0  # 即使较大螺距也给一点偏移
    
    hit, t_hit, pts, rects, pair_hit = first_collision_time_with_offset(
        sample_points, sample_thetas, t_max, t_start)
    
    if hit:
        t_stop = t_hit
        th = head_theta(t_stop)
        r_head = b * th
        entered = r_head <= R_TARGET + 1e-9
        print(f"碰撞@{t_hit:.1f}s, r_head={r_head:.2f}m, 进入={entered}")
        return entered, r_head, t_stop, True, pair_hit
    else:
        # 未碰撞到 t_max，取 t_max 状态
        t_stop = t_max
        th = head_theta(t_stop)
        r_head = b * th
        entered = r_head <= R_TARGET + 1e-9
        print(f"无碰撞@{t_max}s, r_head={r_head:.2f}m, 进入={entered}")
        return entered, r_head, t_stop, False, None

def search_p_min(p0: float = 0.55):
    """改进的网格搜索: 逐步细化找到最小可行螺距
    返回(p_min, feasible, samples)
    """
    samples = []
    
    # 步骤1: 找到一个可行的起始点
    print("步骤1: 评估初始螺距", p0)
    ent, r_head, t_stop, coll, pair = evaluate_pitch(p0)
    samples.append((p0, ent, r_head, t_stop, coll))
    
    if not ent:
        print("初始螺距无法进入，向下搜索可行点...")
        p_current = p0
        while not ent and p_current > 0.05:
            p_current = round(p_current - 0.1, 3)
            ent, r_head, t_stop, coll, pair = evaluate_pitch(p_current)
            samples.append((p_current, ent, r_head, t_stop, coll))
        
        if not ent:
            return p_current, False, samples
        p_feasible = p_current
    else:
        p_feasible = p0
    
    # 步骤2: 逐级网格搜索，找到最大的可行螺距
    step_sizes = [0.1, 0.01, 0.001]
    print(f"步骤2: 逐级网格搜索 (步长: {step_sizes})")
    
    for level, step in enumerate(step_sizes):
        print(f"  级别{level+1}: 步长={step}")
        
        # 在当前级别上，找到最大可行值和最小不可行值之间的边界
        p_min_infeasible = None
        
        # 向上搜索找到第一个不可行点
        p_test = p_feasible + step
        while p_test <= 2.0:  # 合理的上限
            p_test = round(p_test, 3)
            ent, r_head, t_stop, coll, pair = evaluate_pitch(p_test)
            samples.append((p_test, ent, r_head, t_stop, coll))
            
            if ent:
                p_feasible = p_test  # 更新可行边界
                p_test += step
            else:
                p_min_infeasible = p_test  # 找到第一个不可行点
                break
        
        if p_min_infeasible is None:
            # 没有找到不可行边界，当前精度下p_feasible就是答案
            break
        
        # 现在我们有边界: p_feasible (可行) 和 p_min_infeasible (不可行)
        # 在下一级别上，我们只需要在这个小区间内搜索
        
        # 为下一级别准备: 在边界区间内进行精细搜索
        if level < len(step_sizes) - 1:
            next_step = step_sizes[level + 1]
            # 在 [p_feasible, p_min_infeasible] 区间内用更细的步长搜索
            p_test = p_feasible + next_step
            new_feasible = p_feasible
            
            while p_test < p_min_infeasible:
                p_test = round(p_test, 4)
                ent, r_head, t_stop, coll, pair = evaluate_pitch(p_test)
                samples.append((p_test, ent, r_head, t_stop, coll))
                
                if ent:
                    new_feasible = p_test
                    p_test += next_step
                else:
                    break
            
            p_feasible = new_feasible
        
        print(f"    当前可行上界: {p_feasible}")
    
    return round(p_feasible, 3), True, samples

def export_result3(p_min: float, detail_rows, path: str):
    import pandas as pd
    # 验证p_min临界性（反馈要求）
    validation_data = []
    test_values = [p_min - 0.001, p_min, p_min + 0.001]
    for p_test in test_values:
        if p_test >= 0.05:
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
    summary = pd.DataFrame({"key":["p_min", "validation_passed"], 
                          "value":[p_min, validation_data[1]['entered'] and not validation_data[2]['entered']]})
    
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
    p_min, feasible = search_p_min()
    print("p_min=", p_min, "feasible=", feasible)
