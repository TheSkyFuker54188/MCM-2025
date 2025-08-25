import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.integrate import quad

def normalize_vec(x_comp, y_comp):
    """
    归一化向量
    输入: x_comp - x 分量, y_comp - y 分量
    输出: 归一化后的向量 (x_norm, y_norm)
    """
    magnitude = np.sqrt(x_comp**2 + y_comp**2)
    return x_comp / magnitude, y_comp / magnitude

def spiral_radius(angle):
    """
    计算螺旋线的极径
    输入: angle - 角度（弧度制）
    输出: 极径值
    """
    return 8.8 + (0.55 / (2 * np.pi)) * angle

def polar_to_cartesian(angle):
    """
    根据极径和角度计算坐标
    输入: angle - 角度（弧度制）
    输出: 坐标 (x, y)
    """
    radius = spiral_radius(angle)
    x_coord = radius * np.cos(angle)
    y_coord = radius * np.sin(angle)
    return x_coord, y_coord

def velocity_components(angle):
    """
    计算给定角度的速度分量
    输入: angle - 角度（弧度制）
    输出: 速度分量 (vx, vy)
    """
    radius = spiral_radius(angle)
    radius_derivative = 0.55 / (2 * np.pi)
    vx = -radius * np.sin(angle) + radius_derivative * np.cos(angle)
    vy = radius * np.cos(angle) + radius_derivative * np.sin(angle)
    return vx, vy

def export_to_csv(data, filename):
    """
    将数据保存到 CSV 文件
    输入: data - 数据数组, filename - 文件名
    """
    np.savetxt(filename, data, fmt='%.6f', delimiter=',')

def compute_theta_values_and_export():
    """
    计算并导出龙头和龙身的角度值
    """
    global theta_body_values, theta_head_values

    time_steps = np.arange(301)
    theta_head_values = np.array([solve_theta_head(t) for t in time_steps])

    dragon_head_length = 2.86
    body_segment_distance = 1.65
    theta_body_values = np.empty((301, 223))

    for t in range(301):
        current_theta = theta_head_values[t]
        for segment in range(223):
            if segment == 0:
                theta_body_values[t, segment] = next_theta(current_theta, dragon_head_length)
            else:
                theta_body_values[t, segment] = next_theta(current_theta, body_segment_distance)
            current_theta = theta_body_values[t, segment]

    export_to_csv(theta_body_values, 'T1_angle.csv')
    print("角度数据已保存到 T1_angle.csv 文件中。")

def compute_and_save_velocity():
    """
    计算并保存速度数据
    """
    global theta_body_values
    head_speed = 1
    velocity_data = np.empty((301, 224))

    for t in range(301):
        current_speed = head_speed
        velocity_data[t, 0] = current_speed
        for segment in range(222):
            velocity_data[t, segment + 1] = compute_velocity_next(
                theta_body_values[t, segment], theta_body_values[t, segment + 1], current_speed
            )
            current_speed = velocity_data[t, segment + 1]

    export_to_csv(velocity_data, 'T1_velocity.csv')
    print("速度数据已保存到 T1_velocity.csv 文件中。")

def compute_and_save_positions():
    """
    计算并保存位置信息
    """
    combined_theta_values = np.hstack((theta_head_values.reshape(-1, 1), theta_body_values))
    radii = np.vectorize(spiral_radius)(combined_theta_values)

    x_positions = radii * np.cos(combined_theta_values)
    y_positions = radii * np.sin(combined_theta_values)

    column_labels = ['timestamp'] + [f'Point_{i}_{axis}' for i in range(224) for axis in ['x', 'y']]
    timestamps = np.arange(301).reshape(-1, 1)
    combined_positions = np.hstack((timestamps, np.hstack((x_positions, y_positions))))

    position_df = pd.DataFrame(combined_positions, columns=column_labels)
    position_df.to_csv('T1_coordinate.csv', index=False)
    print("位置信息已保存到 T1_coordinate.csv 文件中。")

# 移除 f 和 g 函数逻辑，直接嵌入调用处

def solve_theta_head(t_val):
    """
    解方程 g(x) = t 的 x 值
    输入: t_val - 目标值 t
    输出: 对应的 x 值
    """
    def compute_g_difference(x, target):
        term1 = 0.5 * (x + 32 * np.pi) * np.sqrt((x + 32 * np.pi)**2 + 1) + 0.5 * np.log(np.abs((x + 32 * np.pi) + np.sqrt((x + 32 * np.pi)**2 + 1)))
        term2 = 0.5 * (32 * np.pi) * np.sqrt((32 * np.pi)**2 + 1) + 0.5 * np.log(np.abs((32 * np.pi) + np.sqrt((32 * np.pi)**2 + 1)))
        coefficient = 0.55 / (2 * np.pi)
        return coefficient * (term2 - term1) - target

    initial_guess = np.radians(0)
    try:
        x_val = fsolve(compute_g_difference, x0=initial_guess, args=(t_val))
        return x_val[0]
    except Exception as e:
        print(f"Failed to solve: {e}")
        return None

def g_function(θ2, θ1, b):
    """
    计算方程 g(θ2) - b^2 的值，用于求解 θ2
    输入: theta2 - 待求解的角度 θ2（弧度制）
           theta1 - 已知角度 θ1（弧度制）
           b - 目标值 b
    输出: g(θ2) - a^2 的值
    """
    r1 = spiral_radius(θ1)
    r2 = spiral_radius(θ2)
    g_θ2 = r2**2 - 2 * r1 * r2 * np.cos(θ2 - θ1) + r1**2
    return g_θ2 - b**2

# 定义函数 compute_bench_direction_vector

def compute_bench_direction_vector(θ_n, θ_n1):
    """
    计算板凳的方向向量
    输入: 
        r_n, theta_n - 当前点的极径和角度（弧度制）
        r_n1, theta_n1 - 下一个点的极径和角度（弧度制）
    输出: 
        向量 (lx, ly) 
    """
    r_n = spiral_radius(θ_n)
    r_n1 = spiral_radius(θ_n1)
    lx = r_n * np.cos(θ_n) - r_n1 * np.cos(θ_n1)
    ly = r_n * np.sin(θ_n) - r_n1 * np.sin(θ_n1)
    return lx, ly

# 确保函数定义正确并可用

def solve_theta_head(t_val):
    """
    解方程 g(x) = t 的 x 值
    输入: t_val - 目标值 t
    输出: 对应的 x 值
    """
    def compute_g_difference(x, target):
        term1 = 0.5 * (x + 32 * np.pi) * np.sqrt((x + 32 * np.pi)**2 + 1) + 0.5 * np.log(np.abs((x + 32 * np.pi) + np.sqrt((x + 32 * np.pi)**2 + 1)))
        term2 = 0.5 * (32 * np.pi) * np.sqrt((32 * np.pi)**2 + 1) + 0.5 * np.log(np.abs((32 * np.pi) + np.sqrt((32 * np.pi)**2 + 1)))
        coefficient = 0.55 / (2 * np.pi)
        return coefficient * (term2 - term1) - target

    initial_guess = np.radians(0)
    try:
        x_val = fsolve(compute_g_difference, x0=initial_guess, args=(t_val))
        return x_val[0]
    except Exception as e:
        print(f"Failed to solve: {e}")
        return None

def next_theta(θ1, b):
    """
    使用数值方法计算 θ2，使得 g(θ2) = b^2 且 θ2 - θ1 ∈ (0, 2π)
    输入: theta1 - 已知角度 θ1（弧度制）
           b - 目标值 b
    输出: 满足条件的 θ2 的值
    """
    initial_guesses = [θ1 + 0.2, θ1 + 0.4, θ1 + 0.7, θ1 + 0.9]
    solutions = []
    for guess in initial_guesses:
        θ2_solution = fsolve(g_function, guess, args=(θ1, b))
        θ2 = θ2_solution[0]
        if 0 < (θ2 - θ1) < 2 * np.pi:
            solutions.append(θ2)
    if solutions:
        unique_solutions = np.unique(solutions)
        return min(solutions)
    else:
        return None

def compute_velocity_next(θ_n, θ_n1, u_n):
    """
    计算速度大小 u_{n+1}
    参数：
    u_n: 当前点速度的大小
    v_n: 当前点的速度方向向量
    v_n1: 下一点的速度方向向量
    l: 参考方向的单位向量
    返回：
    u_{n+1}: 下一点的速度大小
    """
    v_n = normalize_vec(*velocity_components(θ_n))
    v_n1 = normalize_vec(*velocity_components(θ_n1))
    l_n_x, l_n_y = compute_bench_direction_vector(θ_n, θ_n1)
    l_n = np.array([l_n_x, l_n_y])
    numerator = np.dot(v_n, l_n)
    denominator = np.dot(v_n1, l_n)
    if denominator == 0:
        raise ValueError("Denominator is zero, cannot divide.")
    u_n1 = (numerator / denominator) * u_n
    return u_n1

def main():
    print("开始执行计算...")

    compute_theta_values_and_export()
    compute_and_save_velocity()
    compute_and_save_positions()

    print("所有计算已完成！")

if __name__ == "__main__":
    main()
