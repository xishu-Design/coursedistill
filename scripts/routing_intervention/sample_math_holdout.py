#!/usr/bin/env python3
"""Create a deterministic MATH holdout subset without external dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "problem" not in row or "answer" not in row:
                raise ValueError(f"{path}:{line_no} must contain problem and answer")
            rows.append(row)
    return rows


def stable_key(row: dict, index: int) -> str:
    return str(row.get("unique_id") or row.get("id") or f"row-{index:05d}")


def stratified_sample(rows: list[dict], size: int, seed: int, field: str) -> list[dict]:
    if size > len(rows):
        raise ValueError(f"Requested {size} rows, but only {len(rows)} are available")
    rng = random.Random(seed)
    groups: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row.get(field, "unknown"))].append((index, row))

    quotas = {}
    fractions = []
    allocated = 0
    for name, group in groups.items():
        exact = size * len(group) / len(rows)
        quotas[name] = min(len(group), int(exact))
        allocated += quotas[name]
        fractions.append((exact - int(exact), rng.random(), name))
    for _, _, name in sorted(fractions, reverse=True):
        if allocated >= size:
            break
        if quotas[name] < len(groups[name]):
            quotas[name] += 1
            allocated += 1

    selected = []
    for name in sorted(groups):
        selected.extend(rng.sample(groups[name], quotas[name]))
    selected.sort(key=lambda item: stable_key(item[1], item[0]))
    return [row for _, row in selected]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--stratify-field", default="subject")
    args = parser.parse_args()

    rows = stratified_sample(read_jsonl(args.input), args.size, args.seed, args.stratify_field)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"SHA256 {digest}")


if __name__ == "__main__":
    main()
