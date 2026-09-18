#!/usr/bin/env python3
"""Generate and compare content-level summaries for selected dataset CSV files.

The semantic hash is based on parsed CSV fields, so it ignores differences in
line endings, UTF-8 BOM handling, delimiter byte representation, and CSV
quoting that do not change field values or row order.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any


TARGET_FILES = (
    "city_load_forecasting/test_data/weather_data/city_1_7_weather_forecast_data.csv",
    "city_load_forecasting/test_data/weather_data/city_16_45_weather_forecast_data.csv",
    "city_load_forecasting/test_data/weather_data/city_8_15_weather_forecast_data.csv",
    "city_load_forecasting/train_data/load_data/city_load_train.csv",
    "pv_load_forecasting/test_data/weather_data/pv_weather_cache.csv",
    "pv_load_forecasting/train_data/load_data/pv_load_train.csv",
)

NULL_VALUES = {"", "null", "none", "nan", "na", "n/a"}


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as source:
        first_line = source.readline()
    if not first_line:
        raise ValueError(f"空文件：{path}")
    return "\t" if first_line.count("\t") > first_line.count(",") else ","


def update_field_hash(digest: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(struct.pack(">Q", len(encoded)))
    digest.update(encoded)


def update_row_hash(digest: Any, row: list[str]) -> None:
    digest.update(struct.pack(">I", len(row)))
    for value in row:
        update_field_hash(digest, value)


def new_column_stats() -> dict[str, Any]:
    return {
        "missing": 0,
        "non_missing": 0,
        "numeric": True,
        "numeric_count": 0,
        "numeric_sum": 0.0,
        "numeric_min": None,
        "numeric_max": None,
        "text_min": None,
        "text_max": None,
    }


def update_column_stats(stats: dict[str, Any], raw_value: str) -> None:
    value = raw_value.strip()
    if value.lower() in NULL_VALUES:
        stats["missing"] += 1
        return

    stats["non_missing"] += 1
    if stats["text_min"] is None or raw_value < stats["text_min"]:
        stats["text_min"] = raw_value
    if stats["text_max"] is None or raw_value > stats["text_max"]:
        stats["text_max"] = raw_value

    try:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError
    except ValueError:
        stats["numeric"] = False
        return

    stats["numeric_count"] += 1
    stats["numeric_sum"] += number
    if stats["numeric_min"] is None or number < stats["numeric_min"]:
        stats["numeric_min"] = number
    if stats["numeric_max"] is None or number > stats["numeric_max"]:
        stats["numeric_max"] = number


def finalize_column_stats(stats: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "missing": stats["missing"],
        "non_missing": stats["non_missing"],
        "text_min": stats["text_min"],
        "text_max": stats["text_max"],
    }
    is_numeric = (
        stats["numeric"]
        and stats["non_missing"] > 0
        and stats["numeric_count"] == stats["non_missing"]
    )
    result["kind"] = "numeric" if is_numeric else "text"
    if is_numeric:
        result.update(
            {
                "min": stats["numeric_min"],
                "max": stats["numeric_max"],
                "mean": stats["numeric_sum"] / stats["numeric_count"],
            }
        )
    return result


def summarize_csv(path: Path, relative_path: str) -> dict[str, Any]:
    delimiter = detect_delimiter(path)
    semantic_digest = hashlib.sha256()
    seen_rows: set[bytes] = set()
    duplicate_rows = 0
    row_count = 0

    with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as source:
        reader = csv.reader(source, delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"空文件：{path}") from exc

        update_row_hash(semantic_digest, header)
        column_stats = [new_column_stats() for _ in header]

        for physical_row, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(
                    f"{relative_path} 第 {physical_row} 行列数为 {len(row)}，"
                    f"表头列数为 {len(header)}"
                )

            row_count += 1
            update_row_hash(semantic_digest, row)

            row_digest = hashlib.sha256()
            update_row_hash(row_digest, row)
            row_key = row_digest.digest()[:16]
            if row_key in seen_rows:
                duplicate_rows += 1
            else:
                seen_rows.add(row_key)

            for stats, value in zip(column_stats, row):
                update_column_stats(stats, value)

    return {
        "path": relative_path,
        "delimiter": "tab" if delimiter == "\t" else "comma",
        "rows": row_count,
        "columns": len(header),
        "header": header,
        "missing_cells": sum(item["missing"] for item in column_stats),
        "duplicate_rows": duplicate_rows,
        "semantic_sha256": semantic_digest.hexdigest(),
        "column_stats": {
            name: finalize_column_stats(stats)
            for name, stats in zip(header, column_stats)
        },
    }


def create_summary(root: Path) -> dict[str, Any]:
    root = root.resolve()
    missing = [relative for relative in TARGET_FILES if not (root / relative).is_file()]
    if missing:
        formatted = "\n".join(f"  - {item}" for item in missing)
        raise FileNotFoundError(f"以下目标文件不存在：\n{formatted}")

    files = []
    for index, relative in enumerate(TARGET_FILES, start=1):
        print(f"[{index}/{len(TARGET_FILES)}] 检查 {relative}", file=sys.stderr)
        files.append(summarize_csv(root / relative, relative))

    return {
        "format_version": 1,
        "root_name": root.name,
        "target_count": len(TARGET_FILES),
        "files": files,
    }


def compare_summaries(left_path: Path, right_path: Path) -> int:
    left = json.loads(left_path.read_text(encoding="utf-8"))
    right = json.loads(right_path.read_text(encoding="utf-8"))
    left_files = {item["path"]: item for item in left["files"]}
    right_files = {item["path"]: item for item in right["files"]}
    all_paths = sorted(left_files.keys() | right_files.keys())
    failures = 0

    for path in all_paths:
        if path not in left_files:
            print(f"MISSING_LEFT  {path}")
            failures += 1
            continue
        if path not in right_files:
            print(f"MISSING_RIGHT {path}")
            failures += 1
            continue

        left_item = left_files[path]
        right_item = right_files[path]
        keys = (
            "rows",
            "columns",
            "header",
            "missing_cells",
            "duplicate_rows",
            "semantic_sha256",
            "column_stats",
        )
        different = [key for key in keys if left_item[key] != right_item[key]]
        if different:
            print(f"DIFFERENT {path}: {', '.join(different)}")
            failures += 1
        else:
            print(f"MATCH     {path}")

    print(
        f"汇总：检查 {len(all_paths)} 个文件，"
        f"一致 {len(all_paths) - failures}，不一致或缺失 {failures}"
    )
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成或比较 CSV 内容摘要（默认排除已验证的天气分片）"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="datasets 根目录（默认脚本所在目录）",
    )
    parser.add_argument("--output", type=Path, help="摘要 JSON 输出路径")
    parser.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("LOCAL_JSON", "SERVER_JSON"),
        help="比较本地与服务器生成的两份摘要",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.compare:
        return compare_summaries(*args.compare)
    if args.output is None:
        raise SystemExit("生成摘要时必须提供 --output")

    summary = create_summary(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"摘要已保存：{args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
