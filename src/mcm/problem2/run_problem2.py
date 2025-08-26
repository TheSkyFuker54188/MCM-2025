import os
from .solver2 import solve_problem2
from .export2 import export_result2

def main():
    hit, t_hit, pts, rects, pair_hit, speeds = solve_problem2(400)
    if not hit:
        print('No collision within range')
        return
    out_path = os.path.abspath('result2.xlsx')
    export_result2(t_hit, pts, speeds, out_path)
    print('Collision at', t_hit, 'exported to', out_path, 'pair', pair_hit)

if __name__ == '__main__':
    main()
