import os
import numpy as np
from .solver import solve_problem1
from .export import export_result1


def validate_physics(times, x, y, speed, vx, vy, theta):
    """验证物理约束：速度一致性、距离保持、运动平滑性"""
    print("开始物理验证...")
    
    # 速度一致性验证
    print("- 验证速度一致性...")
    dt = times[1] - times[0]
    computed_vx = np.gradient(x, dt, axis=0)
    computed_vy = np.gradient(y, dt, axis=0)
    computed_speed = np.sqrt(computed_vx**2 + computed_vy**2)
    
    # 检查速度差异
    speed_diff = np.abs(computed_speed - speed)
    max_speed_diff = np.max(speed_diff)
    print(f"  最大速度差异: {max_speed_diff:.6f} m/s")
    
    vx_diff = np.abs(computed_vx - vx)
    vy_diff = np.abs(computed_vy - vy)
    max_vx_diff = np.max(vx_diff)
    max_vy_diff = np.max(vy_diff)
    print(f"  最大vx差异: {max_vx_diff:.6f} m/s")
    print(f"  最大vy差异: {max_vy_diff:.6f} m/s")
    
    # 距离保持验证（已在solver中实现）
    print("- 距离约束验证已在求解器中实施")
    
    # 运动平滑性验证
    print("- 验证运动平滑性...")
    accel_x = np.gradient(vx, dt, axis=0)
    accel_y = np.gradient(vy, dt, axis=0)
    accel_mag = np.sqrt(accel_x**2 + accel_y**2)
    max_accel = np.max(accel_mag)
    print(f"  最大加速度: {max_accel:.3f} m/s²")
    
    # 角度连续性验证
    print("- 验证角度连续性...")
    theta_diff = np.diff(theta, axis=0)
    # 处理角度跳跃
    theta_diff = np.where(theta_diff > np.pi, theta_diff - 2*np.pi, theta_diff)
    theta_diff = np.where(theta_diff < -np.pi, theta_diff + 2*np.pi, theta_diff)
    max_theta_jump = np.max(np.abs(theta_diff))
    print(f"  最大角度跳跃: {max_theta_jump:.6f} rad")
    
    print("物理验证完成。")
    return {
        'max_speed_diff': max_speed_diff,
        'max_vx_diff': max_vx_diff,
        'max_vy_diff': max_vy_diff,
        'max_accel': max_accel,
        'max_theta_jump': max_theta_jump
    }


def main():
    print("开始求解问题1：基于欧几里得约束的螺旋运动...")
    times, x, y, speed, vx, vy, theta = solve_problem1(300)
    
    # 物理验证
    validation = validate_physics(times, x, y, speed, vx, vy, theta)
    
    # 导出结果
    out_path = os.path.abspath('result1.xlsx')
    export_result1(times, x, y, speed, vx, vy, theta, out_path)
    print(f'结果已写入: {out_path}')
    
    # 总结
    print(f"\n问题1求解完成:")
    print(f"- 时间范围: {times[0]:.1f}s - {times[-1]:.1f}s")
    print(f"- 数据点数: {len(times)}")
    print(f"- 节点数量: {x.shape[1]}")
    print(f"- 最大速度误差: {validation['max_speed_diff']:.6f} m/s")
    print(f"- 最大加速度: {validation['max_accel']:.3f} m/s²")

if __name__ == '__main__':
    main()
