"""
Target definitions for the machine learning signal-calibration layer.
"""

from __future__ import annotations


TARGET_HORIZON_DAYS = 20
TARGET_NAME = "forward_return_20d"
TARGET_COLUMN = f"target_forward_return_{TARGET_HORIZON_DAYS}d"
NEGATIVE_TARGET_COLUMN = f"target_negative_return_{TARGET_HORIZON_DAYS}d"

TARGET_SUMMARY = (
    "The machine learning layer is calibrated to predict forward 20-trading-day return, "
    "which is scale-aware, comparable across assets, and more finance-relevant than "
    "predicting raw price level. Raw prices are non-stationary and dominated by starting "
    "price differences, while forward return better reflects tradable direction and magnitude."
)
