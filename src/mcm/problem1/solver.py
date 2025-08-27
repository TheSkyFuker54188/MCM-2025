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

@njit(cache=True, fastmath=True)
def newton_handle_theta_numba(x_prev: float, y_prev: float, L: float, b: float, theta_guess: float) -> float:
    """Numba版本的牛顿迭代求解把手角度
    
    修复问题：
    1. 迭代直到收敛而不是固定次数
    2. 初始解选取：后面板凳的极角比前面大，所以用 theta_guess + 0.5
    """
    th = theta_guess + 0.5  # 后面板凳的极角应该比前面的大
    
    max_iter = 150  # 最大迭代次数防止无限循环
    tolerance = 1e-12  # 收敛容差
    
    for iteration in range(max_iter):
        r = b * th
        c = math.cos(th); s = math.sin(th)
        X = r * c; Y = r * s
        dx = X - x_prev; dy = Y - y_prev
        f = dx*dx + dy*dy - L*L
        
        # 检查是否收敛
        if abs(f) < tolerance:
            return th
            
        # 计算梯度
        dX = b * c - r * s
        dY = b * s + r * c
        df = 2*dx*dX + 2*dy*dY
        
        # 避免除零
        if abs(df) < 1e-15:
            break
            
        # 牛顿步
        delta = f / df
        th_new = th - delta
        
        # 确保theta保持正值且合理
        if th_new <= 0:
            th_new = th * 0.5  # 如果步长过大导致负值，减半
        elif th_new > th * 3:  # 防止步长过大
            th_new = th * 1.5
            
        th = th_new
        
        # 检查步长是否足够小（另一种收敛判据）
        if abs(delta) < tolerance:
            break
    
    return max(th, 0.0)

def newton_handle_theta(x_prev: float, y_prev: float, L: float, theta_guess: float) -> float:
    """给定上一把手坐标与刚性距离 L, 求当前把手 theta 使 (r cosθ - x_prev)^2 + (r sinθ - y_prev)^2 = L^2.
    r = b θ, 其中 b 由 spiral 模块常量给出 (全局 sp). 直接利用 spiral_pos/其导数。
    使用牛顿迭代; theta_guess 可用上一把手 theta 或略小值。
    """
    from .spiral import sp
    b = sp.b
    return newton_handle_theta_numba(x_prev, y_prev, L, b, theta_guess)


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

        # 2. 其他把手位置 (刚体欧氏约束牛顿)
        for hi in range(1, n_handles):
            L = cp.effective_distance(hi-1)
            theta_i = newton_handle_theta(x[ti, hi-1], y[ti, hi-1], L, theta[ti, hi-1])
            theta[ti, hi] = theta_i
            xi, yi = spiral_pos(theta_i)
            x[ti, hi], y[ti, hi] = xi, yi
            
            # 距离验证断言（反馈要求）
            actual_dist = math.hypot(xi - x[ti, hi-1], yi - y[ti, hi-1])
            if abs(actual_dist - L) > 1e-6:
                print(f"警告: 时刻{t}s 把手{hi} 距离误差 {abs(actual_dist - L):.2e}m (期望{L}m, 实际{actual_dist:.6f}m)")

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

def validate_results(times, x, y, speed, sample_times=[10, 50, 100]):
    """验证计算结果的物理一致性（反馈要求）"""
    print("=== 第一问结果验证 ===")
    cp = ChainParams()
    for t in sample_times:
        if t < len(times):
            print(f"\n时刻 {t}s:")
            # 距离验证
            max_dist_error = 0
            for i in range(1, cp.handle_count):
                expected_L = cp.effective_distance(i-1)
                actual_dist = math.hypot(x[t,i] - x[t,i-1], y[t,i] - y[t,i-1])
                error = abs(actual_dist - expected_L)
                max_dist_error = max(max_dist_error, error)
            print(f"  最大距离误差: {max_dist_error:.2e}m")
            
            # 速度合理性检查
            speeds_t = speed[t, :]
            print(f"  速度范围: {speeds_t.min():.3f} ~ {speeds_t.max():.3f} m/s")
            abnormal_speeds = np.sum((speeds_t < 0) | (speeds_t > 10))
            if abnormal_speeds > 0:
                print(f"  异常速度数量: {abnormal_speeds}")
                
    print("=== 验证完成 ===\n")

if __name__ == "__main__":
    times, x, y, v, vx, vy, theta = solve_problem1(10)
    print("sample x[0,0]", x[0,0])
