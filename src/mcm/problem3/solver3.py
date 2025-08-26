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
from dataclasses import dataclass
from typing import Callable, Tuple, List
import numpy as np

from ..problem1.constants import ChainParams
from ..problem2.collision import first_collision_time

R_TARGET = 4.5
cp = ChainParams()

# ---------- 可变螺距的螺线函数 ----------
def arc_length(theta: float, b: float) -> float:
    return b * 0.5 * (theta * math.sqrt(1 + theta*theta) + math.asinh(theta))

def arc_length_inv(s: float, b: float, theta_guess: float) -> float:
    theta = theta_guess
    for _ in range(25):
        f = arc_length(theta, b) - s
        if abs(f) < 1e-12:
            return theta
        d = b * math.sqrt(1 + theta*theta)
        theta -= f / d
    return theta

def spiral_pos(theta: float, b: float):
    r = b * theta
    return r * math.cos(theta), r * math.sin(theta)

def spiral_tangent_unit(theta: float, b: float):
    dx = b * (math.cos(theta) - theta * math.sin(theta))
    dy = b * (math.sin(theta) + theta * math.cos(theta))
    n = math.hypot(dx, dy)
    return dx / n, dy / n

# 预先把手间距累积
intervals = [cp.distance_between_handles(i) for i in range(cp.n_total)]
cum_offsets = [0.0]
acc = 0.0
for d in intervals:
    acc += d
    cum_offsets.append(acc)

def simulate_handles(T: int, p: float):
    """返回 times, x,y,speed,theta 对于给定螺距 p。"""
    b = p / (2 * math.pi)
    theta_head_0 = 32 * math.pi
    s_head_0 = arc_length(theta_head_0, b)
    v_head = cp.v_head
    n_handles = cp.handle_count
    times = np.arange(0, T+1, dtype=int)
    theta = np.zeros((len(times), n_handles))
    x = np.zeros_like(theta)
    y = np.zeros_like(theta)
    speed = np.zeros_like(theta)

    for ti, t in enumerate(times):
        s_head = s_head_0 - v_head * t
        guess = theta[ti-1,0] if ti>0 else theta_head_0
        th_head = arc_length_inv(s_head, b, guess)
        theta[ti,0] = th_head
        xh,yh = spiral_pos(th_head, b)
        x[ti,0], y[ti,0] = xh,yh
        speed[ti,0] = v_head
        tangents = [spiral_tangent_unit(th_head, b)] + [None]*(n_handles-1)
        # 后续把手位置
        for hi in range(1, n_handles):
            s_i = s_head - cum_offsets[hi]
            if s_i <= 0:
                th = 0.0
            else:
                guess_i = theta[ti, hi-1]
                th = arc_length_inv(s_i, b, guess_i)
            theta[ti, hi] = th
            xi, yi = spiral_pos(th, b)
            x[ti, hi], y[ti, hi] = xi, yi
            tangents[hi] = spiral_tangent_unit(th, b)
        # 速度递推
        for hi in range(1, n_handles):
            dx = x[ti, hi] - x[ti, hi-1]
            dy = y[ti, hi] - y[ti, hi-1]
            dist = math.hypot(dx, dy)
            if dist == 0:
                continue
            ux, uy = dx/dist, dy/dist
            t_prev = tangents[hi-1]
            t_cur = tangents[hi]
            num = t_prev[0]*ux + t_prev[1]*uy
            den = t_cur[0]*ux + t_cur[1]*uy
            if abs(den) < 1e-12:
                continue
            speed[ti, hi] = speed[ti, hi-1] * num / den
    return times, x, y, speed, theta, b

def build_interpolators(T: int, p: float):
    times, x, y, speed, theta, b = simulate_handles(T, p)
    times_f = times.astype(float)
    def sample_points(t: float):
        if t <= times_f[0]:
            idx=0
            return [(x[idx,i], y[idx,i]) for i in range(x.shape[1])]
        if t >= times_f[-1]:
            idx=-1
            return [(x[idx,i], y[idx,i]) for i in range(x.shape[1])]
        k=int(math.floor(t))
        a=t-k
        pts=[]
        for i in range(x.shape[1]):
            pts.append((x[k,i]*(1-a)+x[k+1,i]*a, y[k,i]*(1-a)+y[k+1,i]*a))
        return pts
    def sample_thetas(t: float):
        if t <= times_f[0]:
            return [theta[0,i] for i in range(theta.shape[1]-1)]
        if t >= times_f[-1]:
            return [theta[-1,i] for i in range(theta.shape[1]-1)]
        k=int(math.floor(t)); a=t-k
        vals=[]
        for i in range(theta.shape[1]-1):
            vals.append(theta[k,i]*(1-a)+theta[k+1,i]*a)
        return vals
    def head_theta(t: float):
        if t <= times_f[0]:
            return theta[0,0]
        if t >= times_f[-1]:
            return theta[-1,0]
        k=int(math.floor(t)); a=t-k
        return theta[k,0]*(1-a)+theta[k+1,0]*a
    return sample_points, sample_thetas, head_theta, b

def evaluate_pitch(p: float, t_max: float = 400.0):
    """返回 (entered, r_head_at_stop, t_stop, collision, pair_hit)
    entered: 是否在停止时头部进入 R_TARGET (r<=R_TARGET)
    collision: 是否因碰撞停止 (True) 否则为达到 t_max
    pair_hit: 若碰撞则给出对
    """
    sample_points, sample_thetas, head_theta, b = build_interpolators(int(t_max), p)
    hit, t_hit, pts, rects, pair_hit = first_collision_time(sample_points, sample_thetas, t_max)
    if hit:
        t_stop = t_hit
        th = head_theta(t_stop)
        r_head = b * th
        entered = r_head <= R_TARGET + 1e-9
        return entered, r_head, t_stop, True, pair_hit
    else:
        # 未碰撞到 t_max，取 t_max 状态
        t_stop = t_max
        th = head_theta(t_stop)
        r_head = b * th
        entered = r_head <= R_TARGET + 1e-9
        return entered, r_head, t_stop, False, None

def search_p_min(p_start: float = 0.55, p_min_bound=0.05, p_max_bound=2.0, tol=0.001):
    """搜索临界螺距 p* （记为 p_min 文档要求）: 当 p <= p* 时龙头可进入 (r_head <= R), 当 p > p* 时不能进入。
    返回 (p_star, feasible)
    算法:
      1. 以 p_start 评估进入与否。
      2. 通过向上或向下扩展步长(0.1) 找到一对 (p_enter, p_not) 使得进入性不同并 p_enter < p_not 且 enter(p_enter)=True, enter(p_not)=False。
         若初始与上方都相同则向相反方向扩展; 若遍历边界仍同则报告失败。
      3. 对区间做二分，直到区间长度 < tol。
    注意: 假定单调性 (经验合理)。
    """
    step = 0.1
    entered0, *_ = evaluate_pitch(p_start)
    # 试探方向 (+step)
    p_up = p_start + step
    if p_up > p_max_bound:
        p_up = p_max_bound
    entered_up, *_ = evaluate_pitch(p_up)
    # 若方向同, 尝试向下
    p_down = p_start - step
    if p_down < p_min_bound:
        p_down = p_min_bound
    entered_down, *_ = evaluate_pitch(p_down)

    # 判断哪个方向能触发状态翻转
    if entered0 != entered_up:
        # bracket [min,max]
        if entered0:
            p_enter, p_not = p_start, p_up
        else:
            p_enter, p_not = p_up, p_start
    elif entered0 != entered_down:
        if entered0:
            p_enter, p_not = p_down, p_start
        else:
            p_enter, p_not = p_start, p_down
    else:
        # 扩展向两侧
        p_left = p_down
        p_right = p_up
        entered_left = entered_down
        entered_right = entered_up
        expanded = False
        for _ in range(40):
            # 优先向外扩展较长一侧
            if p_right < p_max_bound:
                p_right = min(p_right + step, p_max_bound)
                entered_right, *_ = evaluate_pitch(p_right)
                if entered_right != entered0:
                    if entered_right:
                        p_enter, p_not = p_right, p_start if not entered0 else p_start
                    else:
                        # entered_right False
                        p_enter = p_start if entered0 else p_left if entered_left else p_left
                        p_not = p_right
                    expanded = True
                    break
            if p_left > p_min_bound:
                p_left = max(p_left - step, p_min_bound)
                entered_left, *_ = evaluate_pitch(p_left)
                if entered_left != entered0:
                    if entered_left:
                        p_enter, p_not = p_left, p_start
                    else:
                        p_enter, p_not = p_start, p_left
                    expanded = True
                    break
            if (p_left <= p_min_bound and p_right >= p_max_bound):
                break
        if not expanded:
            return p_start, False
    # 规范化: ensure p_enter < p_not and enter(p_enter)=True
    if p_enter > p_not:
        p_enter, p_not = p_not, p_enter
    # 保证端点属性
    if not evaluate_pitch(p_enter)[0] or evaluate_pitch(p_not)[0]:
        # 属性不满足, 失败
        return p_enter, False
    # 二分
    left, right = p_enter, p_not
    while right - left > tol:
        mid = (left + right) / 2
        ok, *_ = evaluate_pitch(mid)
        if ok:
            left = mid  # mid 仍可进入, 临界在右侧
        else:
            right = mid
    return round(left, 3), True

def export_result3(p_min: float, detail_rows, path: str):
    import pandas as pd
    df = pd.DataFrame(detail_rows)
    summary = pd.DataFrame({"key":["p_min"], "value":[p_min]})
    with pd.ExcelWriter(path, engine='openpyxl') as w:
        summary.to_excel(w, sheet_name='summary', index=False)
        df.to_excel(w, sheet_name='samples', index=False)

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
