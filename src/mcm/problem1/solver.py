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
            
        dX = b * c - r * s
        dY = b * s + r * c
        df = 2*dx*dX + 2*dy*dY
        
        if abs(df) < 1e-15:
            # 梯度太小，扰动
            if iteration % 2 == 0:
                th = th + 0.01
            else:
                th = th - 0.01
            continue
            
        step = f / df
        # 限制步长
        if abs(step) > 0.3:
            if step > 0:
                step = 0.3
            else:
                step = -0.3
                
        th_new = th - step
        
        if th_new <= 0:
            th_new = th * 0.5
            
        th = th_new
        
    return max(best_th, 0.01)

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
            # 使用改进的numba版本
            from .spiral import sp
            theta_i = newton_handle_theta_numba(x[ti, hi-1], y[ti, hi-1], L, sp.b, theta[ti, hi-1])
            
            # 验证结果合理性
            if theta_i <= 0:
                # 回退策略：基于前一把手角度的保守估计
                theta_i = max(theta[ti, hi-1] * 0.95, 0.1)
                
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
