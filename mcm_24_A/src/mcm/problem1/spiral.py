import math
from typing import Tuple
from .constants import SpiralParams

sp = SpiralParams()

# S(theta) 弧长(从0到theta)
# 推导: ds = b sqrt(1+theta^2) dtheta
# 积分 S(theta)= b/2*( theta*sqrt(1+theta**2) + asinh(theta) )

def spiral_arc_length(theta: float) -> float:
    t = theta
    return sp.b * 0.5 * ( t * math.sqrt(1 + t*t) + math.asinh(t) )

def spiral_arc_length_inv(s: float, theta_guess: float) -> float:
    """给定弧长 s 求 theta，使用牛顿迭代; theta_guess 初始猜测。"""
    # 牛顿迭代
    theta = theta_guess
    for _ in range(20):
        f = spiral_arc_length(theta) - s
        if abs(f) < 1e-12:
            return theta
        # dS/dtheta = b * sqrt(1+theta^2)/2 *2? 实际 ds/dtheta = b*sqrt(1+theta^2)
        d = sp.b * math.sqrt(1 + theta*theta)
        theta -= f / d
    return theta

def spiral_pos(theta: float) -> Tuple[float, float]:
    r = sp.b * theta
    return r * math.cos(theta), r * math.sin(theta)

def spiral_tangent_unit(theta: float) -> Tuple[float, float]:
    # derivative wrt theta: (b(cos - theta sin), b(sin + theta cos))
    dx = sp.b * (math.cos(theta) - theta * math.sin(theta))
    dy = sp.b * (math.sin(theta) + theta * math.cos(theta))
    norm = math.hypot(dx, dy)
    return dx / norm, dy / norm
