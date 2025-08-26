import os
from .solver import solve_problem1
from .export import export_result1


def main():
    times, x, y, speed, vx, vy, theta = solve_problem1(300)
    out_path = os.path.abspath('result1.xlsx')
    export_result1(times, x, y, speed, vx, vy, theta, out_path)
    print('Written', out_path)

if __name__ == '__main__':
    main()
