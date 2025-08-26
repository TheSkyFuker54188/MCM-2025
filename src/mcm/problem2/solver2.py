import math
from typing import Tuple
import numpy as np
from ..problem1.solver import solve_problem1
from .collision import first_collision_time

# 为了复用第一问逻辑，我们先预计算足够长时间的轨迹 (到t_max+1)

def build_interpolators(T_max: int = 400):
    times, x, y, speed, vx, vy, theta = solve_problem1(T_max)
    times_f = times.astype(float)
    def sample_points(t: float):
        # 线性插值
        if t <= times_f[0]:
            idx = 0
            return [(x[idx,i], y[idx,i]) for i in range(x.shape[1])]
        if t >= times_f[-1]:
            idx = -1
            return [(x[idx,i], y[idx,i]) for i in range(x.shape[1])]
        k = int(math.floor(t))
        alpha = t - k
        # clamp
        if k >= times_f[-1]:
            k = int(times_f[-1]) - 1
        idx0 = k
        idx1 = k + 1
        pts = []
        for i in range(x.shape[1]):
            x_interp = x[idx0,i]*(1-alpha) + x[idx1,i]*alpha
            y_interp = y[idx0,i]*(1-alpha) + y[idx1,i]*alpha
            pts.append((x_interp, y_interp))
        return pts
    def sample_thetas(t: float):
        if t <= times_f[0]:
            idx=0
            return [theta[idx,i] for i in range(theta.shape[1]-1)]  # exclude tail rear
        if t >= times_f[-1]:
            idx=-1
            return [theta[idx,i] for i in range(theta.shape[1]-1)]
        k = int(math.floor(t))
        alpha = t - k
        idx0 = k
        idx1 = k + 1
        vals = []
        for i in range(theta.shape[1]-1):
            vals.append(theta[idx0,i]*(1-alpha) + theta[idx1,i]*alpha)
        return vals
    def sample_speeds(t: float):
        # 返回所有把手速度(与points同长度)
        if t <= times_f[0]:
            idx = 0
            return [speed[idx,i] for i in range(speed.shape[1])]
        if t >= times_f[-1]:
            idx = -1
            return [speed[idx,i] for i in range(speed.shape[1])]
        k = int(math.floor(t))
        alpha = t - k
        idx0, idx1 = k, k+1
        vals = []
        for i in range(speed.shape[1]):
            vals.append(speed[idx0,i]*(1-alpha) + speed[idx1,i]*alpha)
        return vals
    return sample_points, sample_thetas, sample_speeds


def solve_problem2(t_max=400):
    sample_points, sample_thetas, sample_speeds = build_interpolators(t_max)
    hit, t_hit, pts, rects, pair_hit = first_collision_time(sample_points, sample_thetas, t_max)
    speeds = None
    if hit:
        speeds = sample_speeds(t_hit)
    return hit, t_hit, pts, rects, pair_hit, speeds

if __name__ == '__main__':
    print(solve_problem2(400))
