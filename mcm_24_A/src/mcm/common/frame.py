"""Common utilities: compute single time frame (positions, theta, speed) given global spiral params.
Uses rigid Euclidean constraints Newton iteration (same as problem1 revised solver).
Designed for sub-second refinement without full time series recomputation.
"""
import math
from typing import Tuple, List
import numpy as np
from ..problem1.constants import ChainParams
from ..problem1.spiral import sp, spiral_arc_length, spiral_arc_length_inv, spiral_pos, spiral_tangent_unit

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def wrap(f):
            return f
        return wrap

cp = ChainParams()

@njit(cache=True, fastmath=True)
def newton_handle_theta_single_numba(x_prev: float, y_prev: float, L: float, b: float, theta_guess: float) -> float:
    th = theta_guess
    if th > 0.5:
        th -= 0.3
    if th < 0:
        th = 0.0
    for _ in range(40):
        r = b * th
        c = math.cos(th); s = math.sin(th)
        X = r * c; Y = r * s
        dx = X - x_prev; dy = Y - y_prev
        f = dx*dx + dy*dy - L*L
        if abs(f) < 1e-11:
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

def newton_handle_theta_single(x_prev: float, y_prev: float, L: float, theta_guess: float) -> float:
    b = sp.b
    return newton_handle_theta_single_numba(x_prev, y_prev, L, b, theta_guess)

def compute_frame(t: float, theta_head_0: float = 32*math.pi, s_head_0: float = None):
    """Compute one frame at real time t (seconds) returning (x_list,y_list,theta_list,speed_list).
    Head moves with constant arc-length speed cp.v_head inward along spiral.
    """
    if s_head_0 is None:
        s_head_0 = spiral_arc_length(theta_head_0)
    v_head = cp.v_head
    s_head = s_head_0 - v_head * t
    # Head theta via inverse arc-length (Newton using previous theta guess ~ head initial)
    theta_head = spiral_arc_length_inv(s_head, theta_head_0)
    xh, yh = spiral_pos(theta_head)
    xs = [xh]; ys=[yh]; thetas=[theta_head]; speeds=[v_head]
    tangents=[spiral_tangent_unit(theta_head)]
    # Subsequent handles rigid distances
    prev_x, prev_y = xh, yh
    prev_theta = theta_head
    for i in range(1, cp.handle_count):
        L = cp.effective_distance(i-1)
        th_i = newton_handle_theta_single(prev_x, prev_y, L, prev_theta)
        xi, yi = spiral_pos(th_i)
        xs.append(xi); ys.append(yi); thetas.append(th_i)
        tangents.append(spiral_tangent_unit(th_i))
        prev_x, prev_y, prev_theta = xi, yi, th_i
    # Velocity propagation (same formula) optional
    for i in range(1, cp.handle_count):
        dx = xs[i] - xs[i-1]; dy = ys[i] - ys[i-1]
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
    return xs, ys, thetas, speeds
