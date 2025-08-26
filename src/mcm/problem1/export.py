import pandas as pd
from .constants import ChainParams

cp = ChainParams()


def export_result1(times, x, y, speed, vx, vy, theta, out_path: str):
    """
    Produce an Excel workbook with two sheets:
    - Positions: rows correspond to labels (龙头x, 龙头y, 第1节龙身x, ... 龙尾(后)y), columns are times `0 s`..`300 s`.
    - Speed: same layout but contains speed magnitudes (m/s).

    All numeric values are written with 6 decimal places.
    """
    n_handles = x.shape[1]
    # build column names
    col_names = [f"{int(t)} s" for t in times]

    # build labels in order described by template
    labels = []
    # head
    labels.append("龙头x (m)")
    labels.append("龙头y (m)")
    # body: 第1节龙身 .. 第221节龙身  -> corresponds to handles 1..221
    for i in range(1, 221 + 1):  # 221 body segments
        labels.append(f"第{i}节龙身x (m)")
        labels.append(f"第{i}节龙身y (m)")
    # tail front handle
    labels.append("龙尾x (m)")
    labels.append("龙尾y (m)")
    # tail rear handle
    labels.append("龙尾（后）x (m)")
    labels.append("龙尾（后）y (m)")

    # prepare data arrays: for each label produce list across times
    data_pos = {name: [] for name in labels}
    data_speed = {name: [] for name in labels}

    # mapping handles: hi 0 -> head, hi 1..221 -> 第1..第221, hi 222 -> tail front, hi 223 -> tail rear
    for ti in range(len(times)):
        # head
        data_pos["龙头x (m)"].append(x[ti, 0])
        data_pos["龙头y (m)"].append(y[ti, 0])
        data_speed["龙头x (m)"].append(speed[ti, 0])
        data_speed["龙头y (m)"].append(speed[ti, 0])

        # body (1..221)
        for bi in range(1, 221 + 1):
            idx = bi  # handle index = bi
            data_pos[f"第{bi}节龙身x (m)"].append(x[ti, idx])
            data_pos[f"第{bi}节龙身y (m)"].append(y[ti, idx])
            data_speed[f"第{bi}节龙身x (m)"].append(speed[ti, idx])
            data_speed[f"第{bi}节龙身y (m)"].append(speed[ti, idx])

        # tail front handle index = cp.n_total - 1? in solver mapping it's at index cp.n_total - 1 +? verify:
        tail_front_idx = cp.n_total - 1  # 222
        tail_rear_idx = cp.n_total  # 223
        data_pos["龙尾x (m)"].append(x[ti, tail_front_idx])
        data_pos["龙尾y (m)"].append(y[ti, tail_front_idx])
        data_pos["龙尾（后）x (m)"].append(x[ti, tail_rear_idx])
        data_pos["龙尾（后）y (m)"].append(y[ti, tail_rear_idx])

        data_speed["龙尾x (m)"].append(speed[ti, tail_front_idx])
        data_speed["龙尾y (m)"].append(speed[ti, tail_front_idx])
        data_speed["龙尾（后）x (m)"].append(speed[ti, tail_rear_idx])
        data_speed["龙尾（后）y (m)"].append(speed[ti, tail_rear_idx])

    # Create matrices
    pos_matrix = [[round(v, 6) for v in data_pos[label]] for label in labels]
    speed_matrix = [[round(v, 6) for v in data_speed[label]] for label in labels]

    df_pos = pd.DataFrame(pos_matrix, columns=col_names)
    df_pos.insert(0, '节点', labels)
    df_speed = pd.DataFrame(speed_matrix, columns=col_names)
    df_speed.insert(0, '节点', labels)

    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        df_pos.to_excel(writer, sheet_name='Positions', index=False)
        df_speed.to_excel(writer, sheet_name='Speed', index=False)
