from .solver3 import search_p_min, export_result3, compute_with_logging
import os

def main():
    p_min, feasible = search_p_min()
    if feasible:
        print(f"最小螺距 p_min = {p_min:.3f} m")
        # 生成邻域验证点
        samples = [p_min + d for d in [-0.005,-0.004,-0.003,-0.002,-0.001,0,0.001,0.002,0.003,0.004,0.005] if p_min + d > 0]
        rows = compute_with_logging(samples)
        out_path = os.path.abspath('result3.xlsx')
        export_result3(p_min, rows, out_path)
        print('result3.xlsx 已生成')
    else:
        print("未找到可行区间, 当前估计 p =", p_min)

if __name__ == '__main__':
    main()
