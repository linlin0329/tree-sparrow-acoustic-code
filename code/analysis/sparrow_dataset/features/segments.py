"""All-segment recording aggregation from matched inputs; no label inference."""

import pandas as pd

GAP = 0.3


def per_recording_allseg(csv):
    d = pd.read_csv(csv, usecols=["source_file", "onset_s", "offset_s", "status"])
    # 用于重建 bout 的"活跃段": 排除 too_short/too_long (过短噪声/过长连续噪声)
    d = d[~d.status.isin(["too_short", "too_long"])].copy()
    rows = []
    for src, g in d.groupby("source_file", sort=False):
        g = g.sort_values("onset_s")
        on = g.onset_s.values.astype(float)
        off = g.offset_s.values.astype(float)
        n = len(on)
        if n == 0:
            rows.append((src, 0, 0, 0.0, 0.0))
            continue
        bouts = [[0]]
        for i in range(1, n):
            if on[i] - off[i - 1] <= GAP:
                bouts[-1].append(i)
            else:
                bouts.append([i])
        bb = max(bouts, key=len)
        dur = float(off[bb[-1]] - on[bb[0]])
        total_active = float((off - on).sum())
        rows.append((src, n, len(bb), dur, total_active))
    return pd.DataFrame(
        rows,
        columns=[
            "source_file",
            "n_seg_all",
            "max_bout_all",
            "bout_dur_all",
            "active_dur_all",
        ],
    )
