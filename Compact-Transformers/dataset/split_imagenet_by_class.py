#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
把本地 ImageNet1k 数据集按类别切分为 front500 与 back500 两份。
假设目录结构如下：
ImageNet1k/
  ├── train/
  │     ├── n01440764/
  │     ├── n01443537/
  │     └── ...
  └── validation/
        ├── n01440764/
        ├── n01443537/
        └── ...

输出：
ImageNet1k_front500/
  ├── train/ ...（前 500 类）
  └── validation/ ...（前 500 类）

ImageNet1k_back500/
  ├── train/ ...（后 500 类）
  └── validation/ ...（后 500 类）
"""

import argparse
import os
from pathlib import Path
import shutil
import sys
import time

def parse_args():
    p = argparse.ArgumentParser(
        description="Split ImageNet1k into front500 and back500 by class (lexicographic)."
    )
    p.add_argument("--src", type=Path, required=True,
                   help="Path to ImageNet1k root (must contain 'train' and 'validation').")
    p.add_argument("--out-root", type=Path, default=None,
                   help="Directory to place the two outputs. Default: same parent as --src.")
    p.add_argument("--method", type=str, default="hardlink",
                   choices=["copy", "hardlink", "symlink"],
                   help="How to create files in the outputs. Default: hardlink.")
    p.add_argument("--dry-run", action="store_true",
                   help="Only print what would happen, without writing files.")
    return p.parse_args()

def ensure_exists(path: Path, dry_run=False):
    if dry_run:
        print(f"[DRY] mkdir -p {path}")
        return
    path.mkdir(parents=True, exist_ok=True)

def list_class_dirs(split_dir: Path):
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    classes = [p.name for p in split_dir.iterdir() if p.is_dir()]
    if not classes:
        raise RuntimeError(f"No class subfolders found under: {split_dir}")
    classes.sort()  # 字典序，保证可复现
    return classes

def choose_method(method: str):
    if method == "copy":
        def op(src, dst):
            shutil.copy2(src, dst)
        return op, "copy"
    elif method == "hardlink":
        def op(src, dst):
            os.link(src, dst)
        return op, "hardlink"
    elif method == "symlink":
        def op(src, dst):
            os.symlink(src, dst)
        return op, "symlink"
    else:
        raise ValueError(f"Unknown method: {method}")

def link_or_copy_file(src: Path, dst: Path, op, method_name: str, dry_run=False):
    if dry_run:
        print(f"[DRY] {method_name} -> {dst}")
        return True

    # 若目标已存在则跳过
    if dst.exists():
        return True

    dst.parent.mkdir(parents=True, exist_ok=True)

    # 首选所选方法；若硬链接/软链接失败则降级为 copy
    try:
        op(src, dst)
        return True
    except Exception as e:
        if method_name in ("hardlink", "symlink"):
            try:
                shutil.copy2(src, dst)
                return True
            except Exception as e2:
                print(f"[ERR] Fallback copy failed: {src} -> {dst} ({e2})", file=sys.stderr)
                return False
        else:
            print(f"[ERR] Copy failed: {src} -> {dst} ({e})", file=sys.stderr)
            return False

def iter_files_recursive(dir_path: Path):
    # 允许类别目录下再有子层级；通常 ImageNet 是平的，但这里做得更稳健
    for p in dir_path.rglob("*"):
        if p.is_file():
            yield p

def transfer_class(src_class_dir: Path, dst_class_dir: Path, op, method_name: str, dry_run=False):
    ok, total = 0, 0
    if not src_class_dir.exists():
        print(f"[WARN] Missing class folder: {src_class_dir}")
        return ok, total
    for f in iter_files_recursive(src_class_dir):
        total += 1
        rel = f.relative_to(src_class_dir)
        dst = dst_class_dir / rel
        if link_or_copy_file(f, dst, op, method_name, dry_run=dry_run):
            ok += 1
    return ok, total

def split_once(src_root: Path, out_root: Path, classes_front, classes_back, split_name: str,
               method_op, method_name: str, dry_run=False):
    src_split = src_root / split_name
    if not src_split.exists():
        raise FileNotFoundError(f"Missing split: {src_split}")

    # 目标根目录
    out_front = out_root / "ImageNet1k_front500" / split_name
    out_back  = out_root / "ImageNet1k_back500"  / split_name
    ensure_exists(out_front, dry_run=dry_run)
    ensure_exists(out_back,  dry_run=dry_run)

    # 处理 front500
    print(f"\n==> [{split_name}] front500 ({len(classes_front)} classes)")
    for cname in classes_front:
        src_c = src_split / cname
        dst_c = out_front / cname
        ensure_exists(dst_c, dry_run=dry_run)
        ok, total = transfer_class(src_c, dst_c, method_op, method_name, dry_run=dry_run)
        if not dry_run:
            print(f"[front500][{split_name}] {cname}: {ok}/{total}")

    # 处理 back500
    print(f"\n==> [{split_name}] back500 ({len(classes_back)} classes)")
    for cname in classes_back:
        src_c = src_split / cname
        dst_c = out_back / cname
        ensure_exists(dst_c, dry_run=dry_run)
        ok, total = transfer_class(src_c, dst_c, method_op, method_name, dry_run=dry_run)
        if not dry_run:
            print(f"[back500][{split_name}] {cname}: {ok}/{total}")

def main():
    args = parse_args()
    src_root: Path = args.src.resolve()
    if args.out_root is None:
        out_root = src_root.parent
    else:
        out_root = args.out_root.resolve()

    # 基本检查
    train_dir = src_root / "train"
    val_dir   = src_root / "validation"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError("Source must contain both 'train' and 'validation' directories.")

    # 获取类别列表（按 train 的类别目录为准，保证一致性）
    classes = list_class_dirs(train_dir)
    if len(classes) != 1000:
        print(f"[WARN] Detected {len(classes)} classes under train/ (expected 1000). Proceeding anyway.")

    # 切分为前 500 与后 500（字典序）
    mid = len(classes) // 2
    classes_front = classes[:mid]
    classes_back  = classes[mid:]

    print(f"Total classes found in train/: {len(classes)}")
    print(f"front500: {len(classes_front)}  back500: {len(classes_back)}")
    if len(classes_front) != len(classes_back):
        print("[WARN] Class count is not 1000; the two splits are not perfectly 500/500.")

    # 选择文件创建方式
    method_op, method_name = choose_method(args.method)
    print(f"File creation method: {method_name}")

    t0 = time.time()

    # 分别处理 train 与 validation
    for split_name in ( "train", "validation"):
        split_once(src_root, out_root, classes_front, classes_back, split_name,
                   method_op, method_name, dry_run=args.dry_run)

    dt = time.time() - t0
    print(f"\nDone in {dt:.1f}s")
    print(f"Outputs:")
    print(f" - { (out_root / 'ImageNet1k_front500').as_posix() }")
    print(f" - { (out_root / 'ImageNet1k_back500').as_posix() }")

if __name__ == "__main__":
    main()
