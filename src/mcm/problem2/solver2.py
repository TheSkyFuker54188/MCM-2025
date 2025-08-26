import math
import numpy as np
from ..problem1.constants import ChainParams
from ..problem1.spiral import spiral_arc_length, spiral_arc_length_inv, spiral_pos, spiral_tangent_unit, sp
from .collision import first_collision_time

try:
    from numba import njit
except ImportError:
    def njit(*a, **k):
        def wrap(f):
            return f
        return wrap

cp = ChainParams()

@njit(cache=True, fastmath=True)
def _newton_handle_theta_numba(x_prev, y_prev, L, b, theta_guess):
    th = theta_guess
    if th < 0:
        th = 0.0
    for _ in range(30):
        r = b * th
        c = math.cos(th); s = math.sin(th)
        X = r * c; Y = r * s
        dx = X - x_prev; dy = Y - y_prev
        f = dx*dx + dy*dy - L*L
        if abs(f) < 1e-12:
            return th
        dX = b * c - r * s
        dY = b * s + r * c
        df = 2*dx*dX + 2*dy*dY
        if df == 0:
            break
        th -= f/df
        if th < 0:
            th = 0.0
    return th

# 预计算刚体距离数组
EFFECTIVE_L = np.array([cp.effective_distance(i) for i in range(cp.handle_count-1)], dtype=np.float64)

theta_head_0 = 32 * math.pi
s_head_0 = spiral_arc_length(theta_head_0)
v_head = cp.v_head

def _compute_frame(t: float, prev_thetas=None):
    """返回 (points(list), thetas(list including tail rear), speeds(list)) 在时间 t 的连续解.
    prev_thetas: 上一时间步的 theta 列表用于初值加速.
    """
    b = sp.b
    # head arc-length -> theta
    s_head = s_head_0 - v_head * t
    guess_head = prev_thetas[0] if prev_thetas is not None else theta_head_0
    theta_head = spiral_arc_length_inv(s_head, guess_head)
    thetas = [theta_head]
    xs = [0.0]; ys = [0.0]
    xh, yh = spiral_pos(theta_head)
    xs[0] = xh; ys[0] = yh
    # other handles
    for i in range(1, cp.handle_count):
        L = EFFECTIVE_L[i-1]
        guess = thetas[-1]  # 邻接初值
        th_i = _newton_handle_theta_numba(xs[i-1], ys[i-1], L, b, guess-0.3)
        x_i, y_i = spiral_pos(th_i)
        thetas.append(th_i)
        xs.append(x_i); ys.append(y_i)
    # 速度递推
    speeds = [v_head]
    tangents = [spiral_tangent_unit(theta_head)]
    for i in range(1, cp.handle_count):
        tangents.append(spiral_tangent_unit(thetas[i]))
        dx = xs[i]-xs[i-1]; dy = ys[i]-ys[i-1]
        dist = math.hypot(dx, dy)
        if dist == 0:
            speeds.append(0.0)
            continue
        ux, uy = dx/dist, dy/dist
        t_prev = tangents[i-1]; t_cur = tangents[i]
        num = t_prev[0]*ux + t_prev[1]*uy
        den = t_cur[0]*ux + t_cur[1]*uy
        if abs(den) < 1e-12:
            speeds.append(0.0)
        else:
            speeds.append(speeds[i-1]*num/den)
    pts = [(xs[i], ys[i]) for i in range(cp.handle_count)]
    return pts, thetas, speeds

def solve_problem2(t_max=1200):
    cache = {}
    last_thetas = None
    def frame_at(t: float):
        key = round(t, 4)
        if key in cache:
            return cache[key]
        nonlocal last_thetas
        frame = _compute_frame(t, last_thetas)
        cache[key] = frame
        last_thetas = frame[1]
        return frame
    def sample_points(t: float):
        return frame_at(t)[0]
    def sample_thetas(t: float):
        return frame_at(t)[1][:-1]
    hit, t_hit, pts, rects, pair_hit = first_collision_time(sample_points, sample_thetas, t_max)
    speeds = frame_at(t_hit)[2] if hit else None
    return hit, t_hit, pts, rects, pair_hit, speeds

if __name__ == '__main__':
    print(solve_problem2(400))
