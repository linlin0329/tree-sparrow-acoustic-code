#!/usr/bin/env python3
"""对完整 30s 原始录音提取 boundary-free 声学特征（songiness 打分器输入）。"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

SR = 32000

from .recording import (  # noqa: E402
    FEATURES,
    extract,
)


def one(path: str) -> dict | None:
    import librosa

    stem = Path(path).stem
    try:
        y, _ = librosa.load(path, sr=SR, mono=True)
        feat = extract(y, SR)
        if feat is None:
            return None
        feat = {key: feat.get(key, np.nan) for key in FEATURES}
        feat["source_file"] = stem
        return feat
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-2023", type=Path, required=True)
    ap.add_argument("--raw-2024", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 1) - 2))
    ap.add_argument("--limit", type=int, default=0, help=">0 时只处理前 N 条（测试）")
    args = ap.parse_args()

    paths = [
        str(path)
        for directory in (args.raw_2023, args.raw_2024)
        for path in sorted(directory.glob("*.mp3"))
    ]
    print(f"音频总数: {len(paths)}")

    done: set[str] = set()
    if args.output_csv.is_file():
        try:
            done = set(
                pd.read_csv(args.output_csv, usecols=["source_file"]).source_file
            )
        except Exception:
            done = set()
    todo = [path for path in paths if Path(path).stem not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"待处理: {len(todo)} (已完成 {len(done)}) | jobs={args.jobs}")
    if not todo:
        print("全部已完成")
        return

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    batch_size = 2000
    header_written = args.output_csv.is_file()
    for i in range(0, len(todo), batch_size):
        chunk = todo[i : i + batch_size]
        results = Parallel(n_jobs=args.jobs, batch_size=8)(
            delayed(one)(path) for path in chunk
        )
        rows = [row for row in results if row is not None]
        if rows:
            frame = pd.DataFrame(rows)
            frame = frame[["source_file"] + FEATURES]
            frame.to_csv(
                args.output_csv,
                mode="a",
                header=not header_written,
                index=False,
            )
            header_written = True
        print(
            f"  {i + len(chunk)}/{len(todo)}  (本批有效 {len(rows)})",
            flush=True,
        )
    print(f"完成 -> {args.output_csv}")


if __name__ == "__main__":
    main()
