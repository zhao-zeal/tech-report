"""Metrics utilities for PV experiments."""
import numpy as np


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def relative_mae(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    return float(np.mean(np.abs(y_true - np.asarray(y_pred)) / (np.abs(y_true) + eps)))


def evaluate(y_true, y_pred):
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "Relative_MAE": relative_mae(y_true, y_pred),
    }
