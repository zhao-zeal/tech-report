#!/usr/bin/env python3
"""Split a CSV into size-limited CSV files, each with the original header."""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path


MB = 1_000_000


def serialized_row(row: list[str], delimiter: str) -> str:
    buffer = io.StringIO(newline="")
    csv.writer(buffer, delimiter=delimiter, lineterminator="\n").writerow(row)
    return buffer.getvalue()


def split_csv(
    input_file: Path,
    output_dir: Path,
    max_size_mb: float = 10,
    encoding: str = "utf-8-sig",
    delimiter: str = ",",
) -> int:
    """Split input_file and return the number of files created."""
    # 使用十进制 MB，默认上限严格为 10,000,000 字节。
    max_bytes = int(max_size_mb * MB)
    if max_bytes <= 0:
        raise ValueError("max_size_mb 必须大于 0")
    if len(delimiter) != 1:
        raise ValueError("delimiter 必须是单个字符")

    output_dir.mkdir(parents=True, exist_ok=True)
    existing_parts = list(output_dir.glob(f"{input_file.stem}_part_*.csv"))
    if existing_parts:
        raise FileExistsError(
            f"输出目录中已有同名分片（例如 {existing_parts[0]}），请换一个空目录，"
            "以免旧分片混入合并结果"
        )
    csv.field_size_limit(min(2**31 - 1, max(csv.field_size_limit(), max_bytes)))

    with input_file.open("r", encoding=encoding, newline="") as source:
        reader = csv.reader(source, delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("输入 CSV 是空文件") from exc

        header_text = serialized_row(header, delimiter)
        header_size = len(header_text.encode("utf-8"))
        if header_size > max_bytes:
            raise ValueError("CSV 表头本身已超过分片大小限制")

        part_number = 0
        part_file = None
        current_size = 0

        def open_part():
            nonlocal part_number, current_size
            part_number += 1
            path = output_dir / f"{input_file.stem}_part_{part_number:04d}.csv"
            handle = path.open("w", encoding="utf-8", newline="")
            handle.write(header_text)
            current_size = header_size
            return handle

        try:
            for row_number, row in enumerate(reader, start=2):
                row_text = serialized_row(row, delimiter)
                row_size = len(row_text.encode("utf-8"))
                if header_size + row_size > max_bytes:
                    raise ValueError(
                        f"第 {row_number} 条 CSV 记录连同表头超过 {max_size_mb:g} MiB，"
                        "无法在不拆坏记录的情况下分片"
                    )

                if part_file is None:
                    part_file = open_part()
                elif current_size + row_size > max_bytes:
                    part_file.close()
                    part_file = open_part()

                part_file.write(row_text)
                current_size += row_size

            # 只有表头时也创建一个有效分片。
            if part_file is None:
                part_file = open_part()
        finally:
            if part_file is not None:
                part_file.close()

    return part_number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按大小拆分 CSV，每个分片均保留表头。")
    parser.add_argument("input", type=Path, help="输入 CSV 文件")
    parser.add_argument("output_dir", type=Path, help="分片输出目录")
    parser.add_argument(
        "--max-size-mb",
        type=float,
        default=10,
        help="每个文件的最大大小（MB；1 MB = 1,000,000 字节，默认 10）",
    )
    parser.add_argument("--encoding", default="utf-8-sig", help="输入编码（默认 utf-8-sig）")
    parser.add_argument("--delimiter", default=",", help="CSV 分隔符（默认逗号）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    count = split_csv(
        args.input,
        args.output_dir,
        args.max_size_mb,
        args.encoding,
        args.delimiter,
    )
    print(f"拆分完成：共生成 {count} 个文件，位置：{args.output_dir.resolve()}")
