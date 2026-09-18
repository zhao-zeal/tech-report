# 2025年春节相位地市用电量回测

该目录用于独立保存本地回测代码、缓存和结果，不修改正式模型、提交结果或 Word 技术报告。

## 评测设置

- 训练截止：2025-01-12。
- 预测窗口：2025-01-13 至 2025-02-09，共 28 天。
- 春节专项：2025-01-27 至 2025-02-04，共 9 天。
- 春节模板：2023、2024 年等权；不使用 2025 年春节电量。
- 气象条件：验证期使用同期实测气象，属于条件回测，不包含预报误差。
- 评分：严格按官方逐城市相对误差 RMSE 得分系数计算，28 天乘 20，专项 9 天乘 5。

2024 年同相位回测起点约为 2024-01-25，起点前历史不足 400 天，因此本轮不构造 2024 完整模型成绩。

## 运行

在 `E:\电量预测赛道\技术报告` 下执行：

```powershell
python .\city_backtest_2025\run_backtest.py
```

首次运行会从 3.6 亿字节历史气象文件生成日级缓存，并训练官方 Prophet Baseline 和截断后的 Ridge。再次运行可复用缓存：

```powershell
python .\city_backtest_2025\run_backtest.py --reuse-cache
```

## 输出

- `results/ablation_results.csv`：官方 Baseline 与 A-H 模块消融汇总。
- `results/weight_sensitivity.csv`：固定分支后的融合权重扫描。
- `results/city_metrics_all_experiments.csv`：全部实验的分城市指标。
- `results/city_comparison_metrics.csv`：Baseline 与完整模型分城市对比及增益。
- `results/daily_predictions.csv`：逐实验、城市、日期预测明细。
- `results/figures/`：300 dpi PNG 与 SVG 图表，包括 10 城市总览和单城市图。
- `results/experiment_summary.md`：中文实验小结与可写入报告的表述。
- `results/validation_checks.json`：时间隔离、覆盖、有限值、权重端点和一致性核验。
- `results/run_manifest.json`：参数、版本、运行命令和关键文件 SHA-256。
- `cache/`：日级气象、Ridge 参数、官方 Baseline 预测及共享分支预测。

脚本读取现有 `baseline/city_pipeline.py` 的 Ridge 特征与训练函数，并在回测层参数化日期、模板年份及消融开关。官方 Baseline 依照 `Baseline.zip` 中 Prophet 参数、节假日/春运构造和四个温度回归量复现。
