import pandas as pd
from typing import List

def _node_names(n_handles: int) -> List[str]:
    # n_handles = 224 (龙头 + 221节龙身 + 龙尾 + 龙尾后)
    # 生成与把手数量一致的节点名称
    # handle 索引 0 -> 龙头
    # 1..221 -> 第i节龙身
    # 222 -> 龙尾
    # 223 -> 龙尾（后）
    names = ["龙头"]
    # 依据题主给出的格式只到第221节龙身
    for i in range(1, 222):
        names.append(f"第{i}节龙身")
    names.append("龙尾")
    names.append("龙尾（后）")
    # 若长度不匹配则截断或填充（理论上应该一致）
    return names[:n_handles]

def export_result2(t_hit: float, pts, speeds, out_path: str):
    """按用户给定格式导出 result2:
    列: 节点, 横坐标x (m), 纵坐标y (m), 速度 (m/s)
    行: 龙头, 第1节龙身 ... 第221节龙身, 龙尾, 龙尾（后）
    另加 meta 工作表记录 collision_time.
    """
    n_handles = len(pts)
    names = _node_names(n_handles)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    df = pd.DataFrame({
        "节点": names,
        "横坐标x (m)": xs,
        "纵坐标y (m)": ys,
        "速度 (m/s)": speeds,
    })
    meta = pd.DataFrame({"key": ["collision_time"], "value": [t_hit]})
    with pd.ExcelWriter(out_path, engine='openpyxl') as w:
        df.to_excel(w, sheet_name='result2', index=False)
        meta.to_excel(w, sheet_name='meta', index=False)
