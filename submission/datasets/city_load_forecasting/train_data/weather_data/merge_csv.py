#!/usr/bin/env python3
"""Merge CSV parts while retaining only one copy of the header."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def merge_csv(
    input_dir: Path,
    output_file: Path,
    pattern: str = "*_part_*.csv",
    encoding: str = "utf-8-sig",
    delimiter: str = ",",
) -> int:
    """Merge matching files in filename order and return the file count."""
    if len(delimiter) != 1:
        raise ValueError("delimiter 必须是单个字符")

    output_resolved = output_file.resolve()
    def natural_key(path: Path) -> list[object]:
        return [int(piece) if piece.isdigit() else piece.lower() for piece in re.split(r"(\d+)", path.name)]

    parts = sorted(
        [
            path
            for path in input_dir.glob(pattern)
            if path.is_file() and path.resolve() != output_resolved
        ],
        key=natural_key,
    )
    if not parts:
        raise FileNotFoundError(f"在 {input_dir} 中未找到匹配 {pattern!r} 的文件")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(2**31 - 1)
    expected_header: list[str] | None = None

    # 先写临时文件，全部成功后再替换目标，避免失败时留下半成品。
    temp_file = output_file.with_name(output_file.name + ".tmp")
    try:
        with temp_file.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.writer(destination, delimiter=delimiter, lineterminator="\n")
            for part in parts:
                with part.open("r", encoding=encoding, newline="") as source:
                    reader = csv.reader(source, delimiter=delimiter)
                    try:
                        header = next(reader)
                    except StopIteration as exc:
                        raise ValueError(f"分片为空：{part}") from exc

                    if expected_header is None:
                        expected_header = header
                        writer.writerow(header)
                    elif header != expected_header:
                        raise ValueError(f"分片表头不一致：{part}")

                    writer.writerows(reader)

        temp_file.replace(output_file)
    except Exception:
        if temp_file.exists():
            temp_file.unlink()
        raise

    return len(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按文件名顺序拼接 CSV 分片。")
    parser.add_argument("input_dir", type=Path, help="分片所在目录")
    parser.add_argument("output", type=Path, help="合并后的 CSV 文件")
    parser.add_argument(
        "--pattern",
        default="*_part_*.csv",
        help="分片匹配模式（默认 *_part_*.csv）",
    )
    parser.add_argument("--encoding", default="utf-8-sig", help="分片编码（默认 utf-8-sig）")
    parser.add_argument("--delimiter", default=",", help="CSV 分隔符（默认逗号）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    count = merge_csv(
        args.input_dir,
        args.output,
        args.pattern,
        args.encoding,
        args.delimiter,
    )
    print(f"拼接完成：已合并 {count} 个文件，输出：{args.output.resolve()}")
