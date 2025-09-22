#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生成：
  class_map_front_full1000_idx2cls.txt
  class_map_back_full1000_idx2cls.txt
  class_map_all_full1000_idx2cls.txt   # <- 新增
  class_filter_front.txt
  class_filter_back.txt
  all_classes_order.txt

数据结构假设：
  /ImageNet1k/
      00000/
      ...
      00999/
"""

import argparse
from pathlib import Path

def scan_classes(root: Path):
    classes = [d.name for d in root.iterdir() if d.is_dir()]
    if not classes:
        raise RuntimeError(f"No class folders found under {root}")
    classes.sort()
    if len(classes) != 1000:
        print(f"[WARN] found {len(classes)} classes (expected 1000). Proceeding anyway.")
    return classes

def write_lines(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(r + "\n")
    print(f"Wrote: {p}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Path to /ImageNet1k (contains 00000..00999)")
    ap.add_argument("--outdir", type=str, default="./class_maps", help="Directory to write outputs")
    ap.add_argument("--front", type=int, default=500, help="Front split size (default 500)")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()

    classes_all = scan_classes(root)
    n = len(classes_all)
    fsz = args.front
    bsz = n - fsz
    assert fsz > 0 and bsz > 0, "front/back split must both be > 0"

    front_classes = classes_all[:fsz]
    back_classes  = classes_all[-bsz:]

    # 过滤清单 + 全顺序
    write_lines(outdir / "class_filter_front.txt", front_classes)
    write_lines(outdir / "class_filter_back.txt", back_classes)
    write_lines(outdir / "all_classes_order.txt", classes_all)

    # full-1000 class_map（前/后专家共用 1000 维，文件名不同方便区分）
    rows_idx2cls = [f"{i}\t{c}" for i, c in enumerate(classes_all)]
    write_lines(outdir / "class_map_front_full1000_idx2cls.txt", rows_idx2cls)
    write_lines(outdir / "class_map_back_full1000_idx2cls.txt", rows_idx2cls)

    # **新增**：完整 1000 类 class_map（统一命名）
    write_lines(outdir / "class_map_all_full1000_idx2cls.txt", rows_idx2cls)

    print("\nDone. Keep class_map_*_full1000 to preserve num_classes=1000; "
          "use class_filter_* to build subset directories for experts.")

if __name__ == "__main__":
    main()
