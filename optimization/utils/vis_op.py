#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import argparse
from pathlib import Path

def extract_values(log_path):
    pattern = re.compile(r"across tasks \[.*?\]: ([0-9.]+)")
    values = []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                values.append(float(match.group(1)))
    return values

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract numbers from log lines like: across tasks ['base', 'taskA', 'taskB']: 0.114..."
    )
    parser.add_argument("--log", type=Path, required=True,
                        help="Path to log file")
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional: save extracted numbers to a text file")

    args = parser.parse_args()

    vals = extract_values(args.log)
    print("Extracted values:", vals)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fout:
            for v in vals:
                fout.write(f"{v}\n")
        print(f"Saved {len(vals)} values to {args.out}")
