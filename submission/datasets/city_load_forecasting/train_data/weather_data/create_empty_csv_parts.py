#!/usr/bin/env python3
"""按 split_csv.py 的命名规则创建空分片文件。"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="创建指定数量的空 CSV 分片文件")
    parser.add_argument(
        "input_name",
        nargs="?",
        default="city_weather_data.csv",
        help="原 CSV 文件名，用于生成分片前缀（默认 city_weather_data.csv）",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=Path("./csv_parts_20M"),
        help="输出目录（默认 ./csv_parts）",
    )
    parser.add_argument("--count", type=int, default=19, help="文件数量（默认 19）")
    args = parser.parse_args()

    if args.count <= 0:
        parser.error("--count 必须大于 0")

    stem = Path(args.input_name).stem
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for number in range(1, args.count + 1):
        output_file = args.output_dir / f"{stem}_part_{number:04d}.csv"
        # 使用独占创建模式，避免意外清空已经存在的分片。
        try:
            output_file.open("x").close()
        except FileExistsError:
            raise SystemExit(f"文件已存在，已停止以避免覆盖：{output_file}")
        print(f"已创建：{output_file}")

    print(f"完成：共创建 {args.count} 个空文件")


if __name__ == "__main__":
    main()
