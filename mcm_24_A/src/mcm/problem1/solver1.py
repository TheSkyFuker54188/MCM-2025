"""兼容入口：保留旧的运行命令 python -m src.mcm.problem1.solver1

该文件仅将调用委托给已有的 run_problem1.main()
"""
from .run_problem1 import main

if __name__ == '__main__':
    main()
