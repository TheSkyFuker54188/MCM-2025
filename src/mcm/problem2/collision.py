import math
from typing import List, Tuple

# 矩形由四顶点顺序给出 (x,y)
Point = Tuple[float, float]
Rect = List[Point]

__all__ = ["build_bench_rects", "first_collision_time"]

# 角点计算
def bench_corners(p_prev: Point, p_next: Point, half_len: float, half_w: float) -> Rect:
    x0, y0 = p_prev
    x1, y1 = p_next
    dx = x1 - x0
    dy = y1 - y0
    dist = math.hypot(dx, dy)
    if dist == 0:
        # 退化
        return [(x0, y0)] * 4
    ux = dx / dist
    uy = dy / dist
    # width normal
    wx = -uy
    wy = ux
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    # 角点
    return [
        (cx + ux * half_len + wx * half_w, cy + uy * half_len + wy * half_w),
        (cx - ux * half_len + wx * half_w, cy - uy * half_len + wy * half_w),
        (cx - ux * half_len - wx * half_w, cy - uy * half_len - wy * half_w),
        (cx + ux * half_len - wx * half_w, cy + uy * half_len - wy * half_w),
    ]

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
    # 策略: 只检查 i 与 j (j>i) 如果 theta_j 在 [theta_i - 3*pi, theta_i) 范围内
    res = []
    for i in range(len(theta_list)):
        ti = theta_list[i]
        low = ti - 3 * math.pi
        for j in range(i + 1, len(theta_list)):
            tj = theta_list[j]
            if low <= tj <= ti:
                res.append((i, j))
    return res


def first_collision_time(sample_func, theta_func, t_max: float, dt_coarse: float = 1.0, refine_levels=(0.1, 0.01, 0.001, 0.0001)):
    """sample_func(t)->points(list of (x,y)), theta_func(t)-> list theta for front handles except tail rear
    返回(是否碰撞, 碰撞时刻, points, 碰撞对)
    """
    t = 0.0
    last_clear = 0.0
    collision_t = None
    pair_hit = None
    while t <= t_max:
        pts, thetas = sample_func(t), theta_func(t)
        rects = build_bench_rects(pts)
        pairs = candidate_pairs(thetas[:len(rects)])
        hit = False
        for a, b in pairs:
            if rects_intersect(rects[a], rects[b]):
                hit = True
                collision_t = t
                pair_hit = (a, b)
                break
        if hit:
            break
        last_clear = t
        t += dt_coarse
    if collision_t is None:
        return False, None, None, None, None
    # refine
    t_start = last_clear
    t_end = collision_t
    for h in refine_levels:
        tt = t_start + h
        found = False
        while tt <= t_end + 1e-12:
            pts, thetas = sample_func(tt), theta_func(tt)
            rects = build_bench_rects(pts)
            pairs = candidate_pairs(thetas[:len(rects)])
            for a, b in pairs:
                if rects_intersect(rects[a], rects[b]):
                    found = True
                    t_end = tt
                    pair_hit = (a, b)
                    break
            if found:
                break
            t_start = tt
            tt += h
    # 最终再在 t_end 采样一次保证位置与矩形对应
    final_pts = sample_func(t_end)
    final_rects = build_bench_rects(final_pts)
    return True, round(t_end, 4), final_pts, final_rects, pair_hit
