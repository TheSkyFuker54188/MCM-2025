import math
from typing import List, Tuple
from ..common.frame import compute_frame

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def wrap(f):
            return f
        return wrap

# 矩形由四顶点顺序给出 (x,y)
Point = Tuple[float, float]
Rect = List[Point]

__all__ = ["build_bench_rects", "first_collision_time"]

@njit(cache=True, fastmath=True)
def bench_corners_numba(x0: float, y0: float, x1: float, y1: float, half_len: float, half_w: float):
    """Numba版本的角点计算"""
    dx = x1 - x0; dy = y1 - y0
    dist = math.sqrt(dx*dx + dy*dy)
    if dist == 0:
        return x0, y0, x0, y0, x0, y0, x0, y0
    ux = dx / dist; uy = dy / dist
    wx = -uy; wy = ux
    cx = (x0 + x1) * 0.5; cy = (y0 + y1) * 0.5
    return (cx + ux * half_len + wx * half_w, cy + uy * half_len + wy * half_w,
            cx - ux * half_len + wx * half_w, cy - uy * half_len + wy * half_w,
            cx - ux * half_len - wx * half_w, cy - uy * half_len - wy * half_w,
            cx + ux * half_len - wx * half_w, cy + uy * half_len - wy * half_w)

# 角点计算
def bench_corners(p_prev: Point, p_next: Point, half_len: float, half_w: float) -> Rect:
    x0, y0 = p_prev; x1, y1 = p_next
    corners = bench_corners_numba(x0, y0, x1, y1, half_len, half_w)
    return [(corners[0], corners[1]), (corners[2], corners[3]), 
            (corners[4], corners[5]), (corners[6], corners[7])]

@njit(cache=True, fastmath=True)
def candidate_pairs_numba(theta_array, n_rects):
    """Numba版本的候选对筛选"""
    pairs = []
    for i in range(n_rects):
        ti = theta_array[i]
        low = ti - 3 * math.pi
        upper = ti - math.pi
        for j in range(0, i):
            tj = theta_array[j]
            if low <= tj <= upper:
                pairs.append((i, j))
    return pairs

# SAT 碰撞检测
def rects_intersect(a: Rect, b: Rect) -> bool:
    def axes(rect: Rect):
        for i in range(4):
            x1, y1 = rect[i]
            x2, y2 = rect[(i + 1) % 4]
            ex = x2 - x1
            ey = y2 - y1
            # 法向量
            nx, ny = -ey, ex
            ln = math.hypot(nx, ny)
            if ln == 0:
                continue
            yield nx / ln, ny / ln
    for ax, ay in list(axes(a)) + list(axes(b)):
        # 投影
        min_a = min(x * ax + y * ay for x, y in a)
        max_a = max(x * ax + y * ay for x, y in a)
        min_b = min(x * ax + y * ay for x, y in b)
        max_b = max(x * ax + y * ay for x, y in b)
        if min_a > max_b or min_b > max_a:
            return False
    return True

# 生成所有矩形
def build_bench_rects(points: List[Point]) -> List[Rect]:
    # points: handles 0..n (tail rear included). Benches: between handle i-1 and i for i=1..n_total
    rects: List[Rect] = []
    # 参数: 第1节半长1.705, 其它1.1, 半宽0.15
    for i in range(1, len(points) - 1):  # last pair before tail rear is real bench tail? treat consistently
        half_len = 1.705 if i == 1 else 1.1
        half_w = 0.15
        rects.append(bench_corners(points[i - 1], points[i], half_len, half_w))
    return rects

# 选择可能碰撞对 (索引基于rects顺序 i=0..n_benches-1)

def candidate_pairs(theta_list: List[float]) -> List[Tuple[int, int]]:
    # 策略修正: 只检查 i 与 j (j>i) 如果 theta_j 在 [theta_i - 3*pi, theta_i - pi] 范围内, 避免相邻干扰
    import numpy as np
    theta_array = np.array(theta_list, dtype=np.float64)
    pairs = candidate_pairs_numba(theta_array, len(theta_list))
    return [(int(p[0]), int(p[1])) for p in pairs]


def first_collision_time(sample_func, theta_func, t_max: float, dt_coarse: float = 1.0):
    """Refined collision search with binary refinement to 1e-4 s using exact frame recomputation.
    sample_func/theta_func kept for compatibility (integer coarse grid), but refinement calls compute_frame.
    """
    t = 0.0
    last_clear = 0.0
    collision_t = None
    pair_hit = None
    while t <= t_max:
        pts, thetas = sample_func(t), theta_func(t)
        rects = build_bench_rects(pts)
        pairs = candidate_pairs(thetas[:len(rects)])
        found=False
        for a,b in pairs:
            if rects_intersect(rects[a], rects[b]):
                collision_t = t
                pair_hit=(a,b)
                found=True
                break
        if found:
            break
        last_clear = t
        t += dt_coarse
    if collision_t is None:
        return False, None, None, None, None
    # Binary search between last_clear and collision_t
    lo = last_clear
    hi = collision_t
    # Precompute head arc-length constant for compute_frame (derive from hi frame sample)
    while hi - lo > 1e-4:
        mid = 0.5*(lo+hi)
        xs, ys, thetas_mid, _ = compute_frame(mid)
        rects_mid = build_bench_rects(list(zip(xs,ys)))
        pairs = candidate_pairs(thetas_mid[:len(rects_mid)])
        collide=False
        for a,b in pairs:
            if rects_intersect(rects_mid[a], rects_mid[b]):
                collide=True
                pair_hit=(a,b)
                break
        if collide:
            hi = mid
        else:
            lo = mid
    final_t = hi
    xs, ys, thetas_f, _ = compute_frame(final_t)
    final_pts = list(zip(xs,ys))
    final_rects = build_bench_rects(final_pts)
    return True, round(final_t,4), final_pts, final_rects, pair_hit
