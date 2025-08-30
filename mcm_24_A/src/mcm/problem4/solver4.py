"""第四问: S型调头路径计算与模拟

实现步骤:
1. 确定边界条件：计算螺线与调头区域的交点，确定入口和出口位置及法向量
2. 推导弧长不变性：证明在给定约束下总弧长唯一
3. 直接求解几何参数：计算两圆弧的半径、圆心、圆心角和总弧长
4. 在复合轨道上进行运动模拟：构建统一的路径参数化函数，初始化模拟状态，执行时间推进循环，计算所有把手的速度
"""
from __future__ import annotations
import math
import numpy as np
from typing import Tuple, Dict, List, Optional, Any, Callable

from ..problem1.constants import ChainParams
from ..problem1.solver import newton_handle_theta, solve_problem1
from ..problem1 import spiral as spiral_module

# 初始化常量
cp = ChainParams()
TURNING_RADIUS = 4.5  # 调头区域半径 (m)
SPIRAL_PITCH = 1.7    # 盘入螺线螺距 (m)
HEAD_SPEED = 1.0      # 龙头前把手沿螺线前进的速度 (m/s)
RADIUS_RATIO = 2.0    # 大圆弧与小圆弧的半径比值 R1:R2 = 2:1

# 尝试导入numba以加速计算
try:
    from numba import njit, jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    jit = njit

def solve_problem4():
    """求解第四问：计算S型调头路径的几何参数并模拟运动过程"""
    # 步骤1：确定边界条件
    entry_point, entry_normal, exit_point, exit_normal = calculate_boundary_conditions()
    
    # 步骤3：直接求解几何参数
    geometry = solve_geometry_parameters(entry_point, entry_normal, exit_point, exit_normal)
    
    # 步骤4：在复合轨道上进行运动模拟
    # 4a：构建统一的路径参数化函数
    path_func = create_path_function(geometry, entry_point, exit_point)
    
    # 4b：初始化模拟状态 (t = -100s)
    initial_state = initialize_simulation(path_func)
    
    # 4c和4d：执行时间推进循环并计算速度
    simulation_results = run_simulation(path_func, initial_state)
    
    return {
        "geometry": geometry,
        "simulation": simulation_results
    }

def calculate_boundary_conditions() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """步骤1：确定边界条件
    
    计算螺线与调头区域的交点，确定入口和出口位置及法向量
    
    返回:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            (入口点坐标, 入口点法向量, 出口点坐标, 出口点法向量)
    """
    # 计算螺线参数 b = p/(2π)
    b = SPIRAL_PITCH / (2 * math.pi)
    
    # 计算螺线与调头区域边界圆的交点
    # 螺线方程：r = b*θ
    # 调头区域边界方程：r = TURNING_RADIUS
    # 联立得：b*θ = TURNING_RADIUS
    # 解得：θ = TURNING_RADIUS / b
    theta_intersection = TURNING_RADIUS / b
    
    print(f"螺线参数 b = {b}")
    print(f"交点极角 θ = {theta_intersection} rad = {math.degrees(theta_intersection)}°")
    
    # 计算入口点的极坐标
    r_entry = TURNING_RADIUS
    theta_entry = theta_intersection
    
    # 转换为直角坐标
    x_entry = r_entry * math.cos(theta_entry)
    y_entry = r_entry * math.sin(theta_entry)
    entry_point = np.array([x_entry, y_entry])
    
    print(f"入口点极坐标: (r={r_entry}, θ={theta_entry})")
    print(f"入口点直角坐标: ({x_entry}, {y_entry})")
    
    # 计算入口点处的切向量 (螺线的切向量)
    # 螺线参数方程：x = b*θ*cos(θ), y = b*θ*sin(θ)
    # 对θ求导得：
    # dx/dθ = b*cos(θ) - b*θ*sin(θ)
    # dy/dθ = b*sin(θ) + b*θ*cos(θ)
    dx_dtheta = b * (math.cos(theta_entry) - theta_entry * math.sin(theta_entry))
    dy_dtheta = b * (math.sin(theta_entry) + theta_entry * math.cos(theta_entry))
    tangent_entry = np.array([dx_dtheta, dy_dtheta])
    tangent_entry = tangent_entry / np.linalg.norm(tangent_entry)  # 单位化
    
    print(f"入口点切向量: ({tangent_entry[0]}, {tangent_entry[1]})")
    
    # 计算入口点处的法向量 (垂直于切向量，指向圆心方向)
    # 法向量应该指向圆心，因此我们需要确保其方向正确
    # 首先计算从入口点指向原点的向量
    to_center = np.array([0, 0]) - entry_point
    to_center = to_center / np.linalg.norm(to_center)  # 单位化
    
    # 计算切向量的垂直向量（两个方向）
    normal1 = np.array([-tangent_entry[1], tangent_entry[0]])
    normal2 = np.array([tangent_entry[1], -tangent_entry[0]])
    
    # 选择与指向原点向量点积为正的那个法向量
    if np.dot(normal1, to_center) > 0:
        normal_entry = normal1
    else:
        normal_entry = normal2
    
    print(f"入口点法向量: ({normal_entry[0]}, {normal_entry[1]})")
    
    # 计算出口点 (与入口点中心对称)
    exit_point = -entry_point
    
    print(f"出口点直角坐标: ({exit_point[0]}, {exit_point[1]})")
    
    # 计算出口点处的切向量 (与入口点切向量中心对称)
    tangent_exit = -tangent_entry
    
    print(f"出口点切向量: ({tangent_exit[0]}, {tangent_exit[1]})")
    
    # 计算出口点处的法向量 (指向圆心方向)
    # 与入口点类似，确保法向量指向原点
    to_center = np.array([0, 0]) - exit_point
    to_center = to_center / np.linalg.norm(to_center)  # 单位化
    
    # 计算切向量的垂直向量（两个方向）
    normal1 = np.array([-tangent_exit[1], tangent_exit[0]])
    normal2 = np.array([tangent_exit[1], -tangent_exit[0]])
    
    # 选择与指向原点向量点积为正的那个法向量
    if np.dot(normal1, to_center) > 0:
        normal_exit = normal1
    else:
        normal_exit = normal2
    
    print(f"出口点法向量: ({normal_exit[0]}, {normal_exit[1]})")
    
    return entry_point, normal_entry, exit_point, normal_exit

def solve_geometry_parameters(
    entry_point: np.ndarray, 
    entry_normal: np.ndarray, 
    exit_point: np.ndarray, 
    exit_normal: np.ndarray
) -> Dict[str, Any]:
    """步骤3：直接求解几何参数
    
    计算两圆弧的半径、圆心、圆心角和总弧长
    
    参数:
        entry_point: 入口点坐标
        entry_normal: 入口点法向量
        exit_point: 出口点坐标
        exit_normal: 出口点法向量
        
    返回:
        Dict[str, Any]: 包含几何参数的字典
    """
    print("入口点:", entry_point)
    print("入口法向量:", entry_normal)
    print("出口点:", exit_point)
    print("出口法向量:", exit_normal)
    
    # 步骤3a：推导求解方程
    # 设定半径比例 k = R1/R2 = 2
    k = RADIUS_RATIO
    
    # 定义常量向量
    delta_P = entry_point - exit_point
    delta_N = k * entry_normal - exit_normal
    
    # 打印中间计算结果以便调试
    print("delta_P:", delta_P)
    print("delta_N:", delta_N)
    
    # 构建一元二次方程 A*R2^2 + B*R2 + C = 0 的系数
    A = np.dot(delta_N, delta_N) - (k + 1)**2
    B = 2 * np.dot(delta_P, delta_N)
    C = np.dot(delta_P, delta_P)
    
    print("方程系数 A:", A)
    print("方程系数 B:", B)
    print("方程系数 C:", C)
    
    # 步骤3b：直接计算半径
    # 使用求根公式解出 R2
    discriminant = B**2 - 4*A*C
    print("判别式:", discriminant)
    
    if discriminant < 0:
        print("警告：判别式小于0，使用直接指定的半径")
        # 如果判别式小于0，我们无法求出实数解
        # 在这种情况下，我们可以直接指定半径
        R2 = TURNING_RADIUS / 3  # 小圆弧半径设为调头区域半径的1/3
        R1 = k * R2  # 大圆弧半径
    else:
        # 计算两个可能的解
        if abs(A) < 1e-10:  # A接近0，方程退化为一次方程
            if abs(B) < 1e-10:  # B也接近0，无法求解
                print("警告：方程系数A和B都接近0，使用直接指定的半径")
                R2 = TURNING_RADIUS / 3
            else:
                R2 = -C / B
        else:
            R2_1 = (-B + math.sqrt(discriminant)) / (2*A)
            R2_2 = (-B - math.sqrt(discriminant)) / (2*A)
            
            print("可能的R2解:", R2_1, R2_2)
            
            # 选择物理上有意义的解（正值）
            if R2_1 > 0:
                R2 = R2_1
            elif R2_2 > 0:
                R2 = R2_2
            else:
                print("警告：无正实数解，使用直接指定的半径")
                R2 = TURNING_RADIUS / 3
        
        # 计算 R1
        R1 = k * R2
    
    print("选择的R2:", R2)
    print("计算的R1:", R1)
    
    # 步骤3c：计算完整几何参数
    # 计算圆心坐标
    C1 = entry_point + R1 * entry_normal
    C2 = exit_point + R2 * exit_normal
    
    print("大圆弧圆心 C1:", C1)
    print("小圆弧圆心 C2:", C2)
    
    # 验证两圆相切条件
    distance_C1_C2 = np.linalg.norm(C1 - C2)
    if not math.isclose(distance_C1_C2, R1 + R2, rel_tol=1e-3):
        print(f"警告：两圆心距离 {distance_C1_C2:.6f} 与半径和 {R1 + R2:.6f} 不相等")
        
        # 调整圆心位置使两圆相切
        C1_to_C2 = C2 - C1
        C1_to_C2_unit = C1_to_C2 / np.linalg.norm(C1_to_C2)
        
        # 调整C2的位置
        C2_new = C1 + (R1 + R2) * C1_to_C2_unit
        
        print("调整后的C2:", C2_new)
        print("调整前后C2的差距:", np.linalg.norm(C2 - C2_new))
        
        # 更新C2
        C2 = C2_new
    
    # 计算两圆弧的连接切点
    C1_to_C2 = C2 - C1
    C1_to_C2_unit = C1_to_C2 / np.linalg.norm(C1_to_C2)
    mid_point = C1 + R1 * C1_to_C2_unit
    
    print("连接切点:", mid_point)
    
    # 计算圆心角
    # 对于大圆弧，计算从入口点到中间点的圆心角
    entry_to_C1 = entry_point - C1
    mid_to_C1 = mid_point - C1
    
    # 归一化向量
    entry_to_C1_unit = entry_to_C1 / np.linalg.norm(entry_to_C1)
    mid_to_C1_unit = mid_to_C1 / np.linalg.norm(mid_to_C1)
    
    # 计算夹角的余弦值
    cos_alpha1 = np.dot(entry_to_C1_unit, mid_to_C1_unit)
    alpha1 = math.acos(min(1.0, max(-1.0, cos_alpha1)))  # 防止浮点误差导致的域错误
    
    # 确定圆心角的符号（顺时针或逆时针）
    # 使用叉积判断旋转方向
    cross_product = np.cross(entry_to_C1_unit, mid_to_C1_unit)
    if cross_product < 0:
        alpha1 = 2 * math.pi - alpha1
    
    # 对于小圆弧，计算从中间点到出口点的圆心角
    mid_to_C2 = mid_point - C2
    exit_to_C2 = exit_point - C2
    
    # 归一化向量
    mid_to_C2_unit = mid_to_C2 / np.linalg.norm(mid_to_C2)
    exit_to_C2_unit = exit_to_C2 / np.linalg.norm(exit_to_C2)
    
    # 计算夹角的余弦值
    cos_alpha2 = np.dot(mid_to_C2_unit, exit_to_C2_unit)
    alpha2 = math.acos(min(1.0, max(-1.0, cos_alpha2)))  # 防止浮点误差导致的域错误
    
    # 确定圆心角的符号
    cross_product = np.cross(mid_to_C2_unit, exit_to_C2_unit)
    if cross_product < 0:
        alpha2 = 2 * math.pi - alpha2
    
    print("大圆弧圆心角 alpha1:", alpha1, "rad =", math.degrees(alpha1), "度")
    print("小圆弧圆心角 alpha2:", alpha2, "rad =", math.degrees(alpha2), "度")
    
    # 计算总弧长
    L_arc1 = alpha1 * R1
    L_arc2 = alpha2 * R2
    L_total = L_arc1 + L_arc2
    
    print("大圆弧弧长 L_arc1:", L_arc1, "m")
    print("小圆弧弧长 L_arc2:", L_arc2, "m")
    print("总弧长 L_total:", L_total, "m")
    
    # 返回几何参数
    return {
        "R1": R1,
        "R2": R2,
        "C1": C1,
        "C2": C2,
        "alpha1": alpha1,
        "alpha2": alpha2,
        "mid_point": mid_point,
        "L_arc1": L_arc1,
        "L_arc2": L_arc2,
        "L_total": L_total
    }

def create_path_function(
    geometry: Dict[str, Any], 
    entry_point: np.ndarray, 
    exit_point: np.ndarray
) -> Callable[[float], Dict[str, np.ndarray]]:
    """步骤4a：构建统一的路径参数化函数
    
    创建一个函数，将全局弧长s映射到轨道上的具体坐标和切线方向
    
    参数:
        geometry: 几何参数字典
        entry_point: 入口点坐标
        exit_point: 出口点坐标
        
    返回:
        Callable[[float], Dict[str, np.ndarray]]: 路径函数，输入弧长s，返回位置和切向量
    """
    # 提取几何参数
    R1 = geometry["R1"]
    R2 = geometry["R2"]
    C1 = geometry["C1"]
    C2 = geometry["C2"]
    alpha1 = geometry["alpha1"]
    alpha2 = geometry["alpha2"]
    mid_point = geometry["mid_point"]
    L_arc1 = geometry["L_arc1"]
    L_total = geometry["L_total"]
    
    # 计算螺线参数 b = p/(2π)
    b = SPIRAL_PITCH / (2 * math.pi)
    
    # 计算入口点的极角
    theta_entry = TURNING_RADIUS / b
    
    # 定义路径函数
    def get_path_info(s: float) -> Dict[str, np.ndarray]:
        """根据全局弧长s获取路径上的点和切向量
        
        参数:
            s: 全局弧长，s=0对应S形曲线的入口切点
            
        返回:
            Dict[str, np.ndarray]: 包含位置和切向量的字典
        """
        if s < 0:
            # 在盘入螺线上
            # 根据弧长s反解出对应的极角theta
            # 使用牛顿迭代法求解方程：spiral_arc_length(theta) = |s|
            # 初始猜测值：基于入口点的极角
            theta_guess = theta_entry + s / (b * theta_entry)
            
            # 使用spiral_arc_length_inv函数求解
            theta = spiral_arc_length_inv(abs(s), b, theta_guess)
            
            # 计算螺线上的位置
            r = b * theta
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            position = np.array([x, y])
            
            # 计算切向量
            dx_dtheta = b * (math.cos(theta) - theta * math.sin(theta))
            dy_dtheta = b * (math.sin(theta) + theta * math.cos(theta))
            tangent = np.array([dx_dtheta, dy_dtheta])
            tangent = tangent / np.linalg.norm(tangent)  # 单位化
            
            return {"position": position, "tangent": tangent}
            
        elif 0 <= s < L_arc1:
            # 在第一段圆弧上（大圆弧）
            # 计算从入口点开始的圆心角
            angle = s / R1
            
            # 计算从入口点到当前点的旋转矩阵
            entry_to_C1 = entry_point - C1
            entry_to_C1_norm = np.linalg.norm(entry_to_C1)
            
            # 确保entry_to_C1是单位向量
            entry_to_C1_unit = entry_to_C1 / entry_to_C1_norm
            
            # 计算垂直于entry_to_C1的单位向量（确保方向正确）
            perp_vector = np.array([-entry_to_C1_unit[1], entry_to_C1_unit[0]])
            
            # 确保perp_vector的方向与圆弧旋转方向一致
            # 如果alpha1 > pi，说明是逆时针旋转
            if alpha1 > math.pi:
                perp_vector = -perp_vector
            
            # 使用旋转公式计算当前位置
            cos_angle = math.cos(angle)
            sin_angle = math.sin(angle)
            rotated_vector = entry_to_C1_unit * cos_angle + perp_vector * sin_angle
            position = C1 + R1 * rotated_vector
            
            # 计算切向量（垂直于半径方向）
            tangent = np.array([-rotated_vector[1], rotated_vector[0]])
            
            # 确保切向量方向正确（沿着圆弧前进方向）
            if alpha1 > math.pi:
                tangent = -tangent
            
            return {"position": position, "tangent": tangent}
            
        elif L_arc1 <= s <= L_total:
            # 在第二段圆弧上（小圆弧）
            # 计算从中间点开始的圆心角
            angle = (s - L_arc1) / R2
            
            # 计算从中间点到当前点的旋转矩阵
            mid_to_C2 = mid_point - C2
            mid_to_C2_norm = np.linalg.norm(mid_to_C2)
            
            # 确保mid_to_C2是单位向量
            mid_to_C2_unit = mid_to_C2 / mid_to_C2_norm
            
            # 计算垂直于mid_to_C2的单位向量（确保方向正确）
            perp_vector = np.array([-mid_to_C2_unit[1], mid_to_C2_unit[0]])
            
            # 确保perp_vector的方向与圆弧旋转方向一致
            # 如果alpha2 > pi，说明是逆时针旋转
            if alpha2 > math.pi:
                perp_vector = -perp_vector
            
            # 使用旋转公式计算当前位置
            cos_angle = math.cos(angle)
            sin_angle = math.sin(angle)
            rotated_vector = mid_to_C2_unit * cos_angle + perp_vector * sin_angle
            position = C2 + R2 * rotated_vector
            
            # 计算切向量（垂直于半径方向）
            tangent = np.array([-rotated_vector[1], rotated_vector[0]])
            
            # 确保切向量方向正确（沿着圆弧前进方向）
            if alpha2 > math.pi:
                tangent = -tangent
            
            return {"position": position, "tangent": tangent}
            
        else:
            # 在盘出螺线上
            # 利用中心对称性质
            # 计算超出S曲线的弧长
            d = s - L_total
            
            # 找到盘入螺线上对称的弧长点
            s_symmetric = -d
            
            # 计算该对称点的信息
            info_symmetric = get_path_info(s_symmetric)
            
            # 盘出螺线上的点是中心对称点
            position = -info_symmetric["position"]
            
            # 切向量同样中心对称
            tangent = -info_symmetric["tangent"]
            
            return {"position": position, "tangent": tangent}
    
    return get_path_info

def spiral_arc_length(theta: float, b: float) -> float:
    """计算螺线从原点到角度theta的弧长
    
    参数:
        theta: 极角
        b: 螺线参数 b = p/(2π)
        
    返回:
        float: 弧长
    """
    # 螺线弧长公式：S(θ) = b/2 * (θ*sqrt(1+θ²) + asinh(θ))
    if theta <= 0:
        return 0.0
    return b * 0.5 * (theta * math.sqrt(1 + theta**2) + math.asinh(theta))

def spiral_arc_length_inv(s: float, b: float, theta_guess: float) -> float:
    """求解给定弧长s对应的极角theta
    
    参数:
        s: 弧长
        b: 螺线参数 b = p/(2π)
        theta_guess: 初始猜测值
        
    返回:
        float: 对应的极角theta
    """
    if s <= 0:
        return 0.0
    
    # 使用牛顿迭代法求解方程：spiral_arc_length(theta) - s = 0
    theta = max(0.1, theta_guess)  # 确保初始值为正
    max_iter = 50
    tolerance = 1e-12
    
    for _ in range(max_iter):
        # 计算函数值和导数
        sqrt_term = math.sqrt(1 + theta**2)
        f = b * 0.5 * (theta * sqrt_term + math.asinh(theta)) - s
        df = b * sqrt_term  # 导数：ds/dθ = b * √(1+θ²)
        
        # 牛顿迭代步骤
        if abs(f) < tolerance:
            break
        
        if abs(df) < 1e-15:  # 避免除以接近零的数
            break
            
        theta = theta - f / df
        
        # 确保theta保持为正值
        if theta <= 0:
            theta = 0.01
    
    return max(0.0, theta)

def initialize_simulation(
    path_func: Callable[[float], Dict[str, np.ndarray]]
) -> Dict[str, Any]:
    """步骤4b：初始化模拟状态 (t = -100s)
    
    计算t = -100s时所有把手的位置
    
    参数:
        path_func: 路径参数化函数
        
    返回:
        Dict[str, Any]: 初始状态
    """
    # 初始时间 t = -100s
    t_initial = -100
    
    # 龙头速度为 1m/s，因此在 t = -100s 时，龙头的全局弧长坐标为 s = 0 - 100 = -100
    s_head = -100.0
    
    # 获取龙头位置和切向量
    head_info = path_func(s_head)
    head_position = head_info["position"]
    head_tangent = head_info["tangent"]
    
    # 初始化所有把手的位置和弧长
    positions = [head_position]  # 把手位置列表，索引0为龙头
    arc_lengths = [s_head]       # 把手弧长列表，索引0为龙头
    tangents = [head_tangent]    # 把手切向量列表，索引0为龙头
    
    # 递推求解后续把手的位置
    for i in range(1, cp.handle_count):
        # 获取前一个把手的位置
        prev_position = positions[i-1]
        
        # 计算当前把手与前一个把手之间的有效距离
        L_i = cp.effective_distance(i-1)
        
        # 使用数值求根算法找到满足距离约束的弧长
        # 初始猜测值：前一个把手的弧长减去有效距离（因为龙头在前进）
        s_guess = arc_lengths[i-1] - L_i
        
        # 定义方程：|position(s) - prev_position|^2 - L_i^2 = 0
        def distance_equation(s):
            pos = path_func(s)["position"]
            return np.sum((pos - prev_position)**2) - L_i**2
        
        # 二分法求根
        s_min = s_guess - 2 * L_i  # 下界
        s_max = arc_lengths[i-1]   # 上界（不能超过前一个把手的弧长）
        
        # 检查边界值是否有效
        f_min = distance_equation(s_min)
        f_max = distance_equation(s_max)
        
        # 确保区间内有根
        if f_min * f_max > 0:
            # 如果边界点函数值符号相同，扩大搜索范围
            s_min = s_guess - 10 * L_i
            f_min = distance_equation(s_min)
            
            if f_min * f_max > 0:
                raise ValueError(f"无法找到把手 {i} 的位置：二分法区间内无根")
        
        # 执行二分法
        tolerance = 1e-9
        max_iter = 50
        
        for _ in range(max_iter):
            s_mid = (s_min + s_max) / 2
            f_mid = distance_equation(s_mid)
            
            if abs(f_mid) < tolerance:
                break
                
            if f_mid * f_min < 0:
                s_max = s_mid
                f_max = f_mid
            else:
                s_min = s_mid
                f_min = f_mid
        
        # 使用最终的弧长计算把手位置和切向量
        s_i = s_mid
        info_i = path_func(s_i)
        position_i = info_i["position"]
        tangent_i = info_i["tangent"]
        
        # 添加到结果列表
        arc_lengths.append(s_i)
        positions.append(position_i)
        tangents.append(tangent_i)
    
    # 计算初始速度
    velocities = calculate_velocities(positions, tangents, HEAD_SPEED)
    
    return {
        "t": t_initial,
        "arc_lengths": arc_lengths,
        "positions": positions,
        "tangents": tangents,
        "velocities": velocities
    }

def calculate_velocities(
    positions: List[np.ndarray], 
    tangents: List[np.ndarray], 
    head_speed: float
) -> List[np.ndarray]:
    """计算所有把手的速度
    
    使用第一问修正后的速度递推公式
    
    参数:
        positions: 所有把手的位置
        tangents: 所有把手的切向量
        head_speed: 龙头速度
        
    返回:
        List[np.ndarray]: 所有把手的速度向量
    """
    # 龙头速度已知
    v_magnitudes = [head_speed]  # 速度大小
    v_vectors = [head_speed * tangents[0]]  # 速度向量
    
    # 递推计算后续把手的速度
    for i in range(1, len(positions)):
        # 计算板凳单位向量
        bench_vector = positions[i-1] - positions[i]
        bench_length = np.linalg.norm(bench_vector)
        
        if bench_length < 1e-10:  # 避免除以零
            bench_unit = np.array([0.0, 0.0])
        else:
            bench_unit = bench_vector / bench_length
        
        # 计算速度比例因子
        dot_prev = np.dot(tangents[i-1], bench_unit)
        dot_curr = np.dot(tangents[i], bench_unit)
        
        # 避免除以接近零的值
        if abs(dot_curr) < 1e-10:
            v_i = v_magnitudes[i-1]  # 如果分母接近零，假设速度不变
        else:
            v_i = v_magnitudes[i-1] * (dot_prev / dot_curr)
        
        # 添加到结果列表
        v_magnitudes.append(v_i)
        v_vectors.append(v_i * tangents[i])
    
    return v_vectors

def run_simulation(
    path_func: Callable[[float], Dict[str, np.ndarray]], 
    initial_state: Dict[str, Any]
) -> Dict[str, Any]:
    """步骤4c和4d：执行时间推进循环并计算速度
    
    从t = -99s到t = 100s，计算所有把手的位置和速度
    
    参数:
        path_func: 路径参数化函数
        initial_state: 初始状态
        
    返回:
        Dict[str, Any]: 模拟结果
    """
    # 提取初始状态
    t_start = initial_state["t"]
    
    # 创建结果容器
    times = [t_start]
    all_positions = [initial_state["positions"]]
    all_arc_lengths = [initial_state["arc_lengths"]]
    all_tangents = [initial_state["tangents"]]
    all_velocities = [initial_state["velocities"]]
    
    # 设置时间步长
    dt = 1.0  # 1秒
    
    # 从t = -99s到t = 100s的时间循环
    for t in range(t_start + 1, 101):
        if t % 20 == 0:
            print(f"模拟进度: t = {t}s")
        
        # 更新龙头位置
        # 龙头的弧长位置每次增加1米（因为速度是1m/s）
        s_head = all_arc_lengths[-1][0] + HEAD_SPEED * dt
        
        # 获取龙头新位置和切向量
        head_info = path_func(s_head)
        head_position = head_info["position"]
        head_tangent = head_info["tangent"]
        
        # 初始化当前时刻的位置、弧长和切向量列表
        positions = [head_position]
        arc_lengths = [s_head]
        tangents = [head_tangent]
        
        # 递推求解后续把手的位置
        for i in range(1, cp.handle_count):
            # 获取前一个把手的位置
            prev_position = positions[i-1]
            
            # 计算当前把手与前一个把手之间的有效距离
            L_i = cp.effective_distance(i-1)
            
            # 使用前一时刻的弧长作为初始猜测值
            s_guess = all_arc_lengths[-1][i]
            
            # 定义距离约束方程
            def distance_equation(s):
                pos = path_func(s)["position"]
                return np.sum((pos - prev_position)**2) - L_i**2
            
            # 二分法求根
            # 设置搜索区间：以上一时刻的弧长为中心，向两侧扩展
            search_range = max(5.0, L_i * 2)
            s_min = s_guess - search_range
            s_max = s_guess + search_range
            
            # 检查边界值是否有效
            f_min = distance_equation(s_min)
            f_max = distance_equation(s_max)
            
            # 确保区间内有根
            if f_min * f_max > 0:
                # 如果边界点函数值符号相同，扩大搜索范围
                s_min = s_guess - search_range * 5
                s_max = s_guess + search_range * 5
                f_min = distance_equation(s_min)
                f_max = distance_equation(s_max)
                
                if f_min * f_max > 0:
                    # 如果仍然无法找到根，使用更保守的方法：
                    # 假设把手沿着路径向前或向后移动
                    # 尝试从当前位置向前搜索
                    s_try = s_guess
                    max_steps = 100
                    step_size = L_i * 0.1
                    found = False
                    
                    for _ in range(max_steps):
                        s_try += step_size
                        f_try = distance_equation(s_try)
                        if abs(f_try) < 1e-6:
                            s_mid = s_try
                            found = True
                            break
                    
                    if not found:
                        # 尝试从当前位置向后搜索
                        s_try = s_guess
                        for _ in range(max_steps):
                            s_try -= step_size
                            f_try = distance_equation(s_try)
                            if abs(f_try) < 1e-6:
                                s_mid = s_try
                                found = True
                                break
                    
                    if not found:
                        # 如果仍然找不到解，使用上一时刻的弧长
                        print(f"警告: 在t={t}s时无法找到把手{i}的位置，使用上一时刻的弧长")
                        s_mid = s_guess
                else:
                    # 执行二分法
                    tolerance = 1e-9
                    max_iter = 50
                    s_mid = (s_min + s_max) / 2
                    
                    for _ in range(max_iter):
                        s_mid = (s_min + s_max) / 2
                        f_mid = distance_equation(s_mid)
                        
                        if abs(f_mid) < tolerance:
                            break
                            
                        if f_mid * f_min < 0:
                            s_max = s_mid
                            f_max = f_mid
                        else:
                            s_min = s_mid
                            f_min = f_mid
            else:
                # 执行二分法
                tolerance = 1e-9
                max_iter = 50
                s_mid = (s_min + s_max) / 2
                
                for _ in range(max_iter):
                    s_mid = (s_min + s_max) / 2
                    f_mid = distance_equation(s_mid)
                    
                    if abs(f_mid) < tolerance:
                        break
                        
                    if f_mid * f_min < 0:
                        s_max = s_mid
                        f_max = f_mid
                    else:
                        s_min = s_mid
                        f_min = f_mid
            
            # 使用最终的弧长计算把手位置和切向量
            s_i = s_mid
            info_i = path_func(s_i)
            position_i = info_i["position"]
            tangent_i = info_i["tangent"]
            
            # 添加到结果列表
            arc_lengths.append(s_i)
            positions.append(position_i)
            tangents.append(tangent_i)
        
        # 计算当前时刻的速度
        velocities = calculate_velocities(positions, tangents, HEAD_SPEED)
        
        # 将当前时刻的结果添加到总结果中
        times.append(t)
        all_positions.append(positions)
        all_arc_lengths.append(arc_lengths)
        all_tangents.append(tangents)
        all_velocities.append(velocities)
    
    # 返回模拟结果
    return {
        "times": times,
        "positions": all_positions,
        "arc_lengths": all_arc_lengths,
        "tangents": all_tangents,
        "velocities": all_velocities
    }

def export_result4(result: Dict[str, Any], path: str):
    """导出第四问结果到Excel文件
    
    参数:
        result: 求解结果
        path: 输出文件路径
    """
    import pandas as pd
    from openpyxl import Workbook
    from openpyxl.utils.dataframe import dataframe_to_rows
    
    # 提取几何参数
    geometry = result["geometry"]
    R1 = geometry["R1"]
    R2 = geometry["R2"]
    C1 = geometry["C1"]
    C2 = geometry["C2"]
    alpha1 = geometry["alpha1"]
    alpha2 = geometry["alpha2"]
    L_total = geometry["L_total"]
    
    # 提取模拟结果
    simulation = result["simulation"]
    times = simulation["times"]
    all_positions = simulation["positions"]
    all_velocities = simulation["velocities"]
    
    # 创建Excel工作簿
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        # 1. 创建几何参数表
        geometry_data = {
            "参数": ["大圆弧半径 R1", "小圆弧半径 R2", 
                   "大圆弧圆心 C1_x", "大圆弧圆心 C1_y", 
                   "小圆弧圆心 C2_x", "小圆弧圆心 C2_y", 
                   "大圆弧圆心角 alpha1 (rad)", "大圆弧圆心角 alpha1 (deg)", 
                   "小圆弧圆心角 alpha2 (rad)", "小圆弧圆心角 alpha2 (deg)", 
                   "总弧长 L"],
            "数值": [R1, R2, 
                   C1[0], C1[1], 
                   C2[0], C2[1], 
                   alpha1, math.degrees(alpha1), 
                   alpha2, math.degrees(alpha2), 
                   L_total]
        }
        geometry_df = pd.DataFrame(geometry_data)
        geometry_df.to_excel(writer, sheet_name='几何参数', index=False)
        
        # 2. 创建关键时刻的位置表（t = -100, 0, 100）
        key_times = [-100, 0, 100]
        key_indices = [times.index(t) for t in key_times]
        
        for idx, t in zip(key_indices, key_times):
            positions = all_positions[idx]
            velocities = all_velocities[idx]
            
            # 创建位置数据
            positions_data = []
            for i in range(min(cp.handle_count, len(positions))):
                pos = positions[i]
                vel = velocities[i]
                vel_magnitude = np.linalg.norm(vel)
                
                positions_data.append({
                    "把手索引": i,
                    "x坐标 (m)": pos[0],
                    "y坐标 (m)": pos[1],
                    "速度x分量 (m/s)": vel[0],
                    "速度y分量 (m/s)": vel[1],
                    "速度大小 (m/s)": vel_magnitude
                })
            
            positions_df = pd.DataFrame(positions_data)
            positions_df.to_excel(writer, sheet_name=f't={t}s', index=False)
        
        # 3. 创建龙头轨迹表
        head_trajectory = []
        for t_idx, t in enumerate(times):
            head_pos = all_positions[t_idx][0]
            head_vel = all_velocities[t_idx][0]
            
            head_trajectory.append({
                "时间 (s)": t,
                "x坐标 (m)": head_pos[0],
                "y坐标 (m)": head_pos[1],
                "速度x分量 (m/s)": head_vel[0],
                "速度y分量 (m/s)": head_vel[1],
                "速度大小 (m/s)": np.linalg.norm(head_vel)
            })
        
        head_trajectory_df = pd.DataFrame(head_trajectory)
        head_trajectory_df.to_excel(writer, sheet_name='龙头轨迹', index=False)
        
        # 4. 创建全部数据表（每20秒采样一次，节省空间）
        sample_indices = [i for i, t in enumerate(times) if t % 20 == 0 or t in [-100, 0, 100]]
        sample_indices.sort()  # 确保按时间顺序
        
        all_data = []
        for t_idx in sample_indices:
            t = times[t_idx]
            positions = all_positions[t_idx]
            velocities = all_velocities[t_idx]
            
            for i in range(min(cp.handle_count, len(positions))):
                pos = positions[i]
                vel = velocities[i]
                
                all_data.append({
                    "时间 (s)": t,
                    "把手索引": i,
                    "x坐标 (m)": pos[0],
                    "y坐标 (m)": pos[1],
                    "速度x分量 (m/s)": vel[0],
                    "速度y分量 (m/s)": vel[1],
                    "速度大小 (m/s)": np.linalg.norm(vel)
                })
        
        all_data_df = pd.DataFrame(all_data)
        all_data_df.to_excel(writer, sheet_name='全部数据', index=False)
    
    print(f"结果已导出到 {path}")

if __name__ == "__main__":
    result = solve_problem4()
    print("S型调头路径几何参数:")
    print(f"  大圆弧半径 R1 = {result['geometry']['R1']:.4f} m")
    print(f"  小圆弧半径 R2 = {result['geometry']['R2']:.4f} m")
    print(f"  大圆弧圆心 C1 = ({result['geometry']['C1'][0]:.4f}, {result['geometry']['C1'][1]:.4f})")
    print(f"  小圆弧圆心 C2 = ({result['geometry']['C2'][0]:.4f}, {result['geometry']['C2'][1]:.4f})")
    print(f"  大圆弧圆心角 α1 = {result['geometry']['alpha1']:.4f} rad = {math.degrees(result['geometry']['alpha1']):.2f}°")
    print(f"  小圆弧圆心角 α2 = {result['geometry']['alpha2']:.4f} rad = {math.degrees(result['geometry']['alpha2']):.2f}°")
    print(f"  总弧长 L = {result['geometry']['L_total']:.4f} m")
    
    export_result4(result, "result4.xlsx")
    print("结果已导出到 result4.xlsx") 