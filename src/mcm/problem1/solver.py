import math
from typing import Tuple
import numpy as np
from .constants import ChainParams
from .spiral import spiral_arc_length, spiral_arc_length_inv, spiral_pos, spiral_tangent_unit

try:
    from numba import njit
except ImportError:  # graceful fallback
    def njit(*args, **kwargs):
        def wrap(f):
            return f
        return wrap

cp = ChainParams()

# 预先计算每个把手到龙头前把手的弧长距离 (沿中心线顺序), 用于快速定位theta
# handle 0 = head front
# handle i 与 handle i-1 之间距离为 distance_between_handles(i-1)
intervals = [cp.distance_between_handles(i) for i in range(cp.n_total)]  # len n_total
cum_handle_s_offsets = [0.0]
acc = 0.0
for d in intervals:
    acc += d
    cum_handle_s_offsets.append(acc)
# 长度 = n_total+1 = handle_count


def solve_problem1(T: int = 300):
    # 初始龙头角度: theta0 = 32 pi
    theta_head_0 = 32 * math.pi
    s_head_0 = spiral_arc_length(theta_head_0)
    v_head = cp.v_head  # constant arc-length speed

    n_handles = cp.handle_count
    times = np.arange(0, T + 1, 1, dtype=int)

    # 结果数组
    theta = np.zeros((len(times), n_handles), dtype=float)
    x = np.zeros_like(theta)
    y = np.zeros_like(theta)
    speed = np.zeros_like(theta)
    vx = np.zeros_like(theta)
    vy = np.zeros_like(theta)

    # 先求所有时刻龙头theta (用弧长倒推) s = s0 + v*t
    for ti, t in enumerate(times):
        # 1. 龙头
        s_head = s_head_0 - v_head * t
        guess = theta[ti-1, 0] if ti > 0 else theta_head_0
        theta_head = spiral_arc_length_inv(s_head, guess)
        theta[ti, 0] = theta_head
        x_head, y_head = spiral_pos(theta_head)
        x[ti, 0], y[ti, 0] = x_head, y_head
        speed[ti, 0] = v_head
        tx0, ty0 = spiral_tangent_unit(theta_head)
        vx[ti, 0] = v_head * tx0
        vy[ti, 0] = v_head * ty0

        # 2. 其他把手位置
        for hi in range(1, n_handles):
            s_i = s_head - cum_handle_s_offsets[hi]
            if s_i <= 0:
                theta_i = 0.0
            else:
                guess_i = theta[ti, hi-1]
                theta_i = spiral_arc_length_inv(s_i, guess_i)
            theta[ti, hi] = theta_i
            xi, yi = spiral_pos(theta_i)
            x[ti, hi], y[ti, hi] = xi, yi

        # 3. 速度递推
        tangents = [spiral_tangent_unit(theta[ti, h]) for h in range(n_handles)]
        for hi in range(1, n_handles):
            dx = x[ti, hi] - x[ti, hi-1]
            dy = y[ti, hi] - y[ti, hi-1]
            dist = math.hypot(dx, dy)
            if dist == 0:
                speed[ti, hi] = 0.0
                vx[ti, hi] = 0.0
                vy[ti, hi] = 0.0
                continue
            ux, uy = dx / dist, dy / dist
            t_prev = tangents[hi - 1]
            t_cur = tangents[hi]
            numerator = t_prev[0] * ux + t_prev[1] * uy
            denominator = t_cur[0] * ux + t_cur[1] * uy
            if abs(denominator) < 1e-12:
                speed[ti, hi] = 0.0
                vx[ti, hi] = 0.0
                vy[ti, hi] = 0.0
            else:
                speed[ti, hi] = speed[ti, hi-1] * numerator / denominator
                vx[ti, hi] = speed[ti, hi] * t_cur[0]
                vy[ti, hi] = speed[ti, hi] * t_cur[1]

    return times, x, y, speed, vx, vy, theta
    return times, x, y, speed, vx, vy

if __name__ == "__main__":
    times, x, y, v, vx, vy, theta = solve_problem1(10)
    print("sample x[0,0]", x[0,0])
