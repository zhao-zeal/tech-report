# PV Analysis Experiments

This directory contains post-submission analysis experiments for the PV forecasting pipeline.

## Goal

Analyze the contribution of the final PV fusion pipeline without modifying the official submission code.

Current target pipeline:

- V36 component
- V48 component
- Fixed fusion: 0.75 * V36 + 0.25 * V48

## Experiment Plan

### 1. Component Ablation

Compare:

- V36 only
- V48 only
- Full fusion

Purpose:

Evaluate the contribution of each forecasting branch.

### 2. Fusion Weight Sensitivity

Keep component predictions fixed and scan V48 fusion weight:

```
0.0, 0.1, 0.2, 0.25, 0.4, 0.6, 0.8, 1.0
```

Purpose:

Verify the robustness of the selected fusion ratio.

### 3. Error Analysis

Analyze PV-user-level prediction errors:

- MAE
- RMSE
- score contribution
- improvement compared with baseline

## Principles

- Do not modify `submission/` production code.
- Keep experiments reproducible.
- Use fixed data split and evaluation protocol.
- Save configurations and results.
