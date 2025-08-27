"""第四问主运行脚本：计算S型调头路径并模拟运动过程"""
import os
from .solver4 import solve_problem4, export_result4

def main():
    """运行第四问求解并导出结果"""
    print("开始求解第四问...")
    
    result = solve_problem4()
    
    # 打印关键结果
    print("\nS型调头路径几何参数:")
    print(f"  大圆弧半径 R1 = {result['geometry']['R1']:.4f} m")
    print(f"  小圆弧半径 R2 = {result['geometry']['R2']:.4f} m")
    print(f"  大圆弧圆心 C1 = ({result['geometry']['C1'][0]:.4f}, {result['geometry']['C1'][1]:.4f})")
    print(f"  小圆弧圆心 C2 = ({result['geometry']['C2'][0]:.4f}, {result['geometry']['C2'][1]:.4f})")
    print(f"  大圆弧圆心角 α1 = {result['geometry']['alpha1']:.4f} rad")
    print(f"  小圆弧圆心角 α2 = {result['geometry']['alpha2']:.4f} rad")
    print(f"  总弧长 L = {result['geometry']['L_total']:.4f} m")
    
    # 导出结果
    out_path = os.path.abspath('result4.xlsx')
    export_result4(result, out_path)
    print(f"\n结果已导出到 {out_path}")

if __name__ == "__main__":
    main() 