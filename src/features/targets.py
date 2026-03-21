"""
Target construction for forward return and downside-risk forecasting.
"""

from __future__ import annotations

import pandas as pd


def build_forward_return_targets(
    price_history: pd.DataFrame,
    horizon_days: int = 20,
) -> pd.DataFrame:
    """Build forward return and negative-return classification targets."""

    if price_history.empty:
        return pd.DataFrame(
            columns=[
                "asset_id",
                "ticker",
                "feature_date",
                f"target_forward_return_{horizon_days}d",
                f"target_negative_return_{horizon_days}d",
            ]
        )

    frame = price_history.copy()
    price_column = (
        "analysis_price"
        if "analysis_price" in frame.columns
        else "adjusted_close"
        if "adjusted_close" in frame.columns
        else "close_price"
    )
    frame["price_date"] = pd.to_datetime(frame["price_date"])
    frame[price_column] = pd.to_numeric(frame[price_column], errors="coerce")
    frame = frame.dropna(subset=[price_column]).sort_values(["asset_id", "price_date"])

    target_frames: list[pd.DataFrame] = []
    forward_return_column = f"target_forward_return_{horizon_days}d"
    negative_return_column = f"target_negative_return_{horizon_days}d"

    for (_, _), group in frame.groupby(["asset_id", "ticker"], sort=False):
        asset_frame = group.copy()
        current_price = asset_frame[price_column].astype(float)
        future_price = current_price.shift(-horizon_days)
        forward_return = future_price / current_price - 1.0

        asset_frame["feature_date"] = asset_frame["price_date"].dt.date
        asset_frame[forward_return_column] = forward_return
        asset_frame[negative_return_column] = (forward_return < 0.0).astype("float")
        asset_frame.loc[forward_return.isna(), negative_return_column] = pd.NA

        target_frames.append(
            asset_frame[
                [
                    "asset_id",
                    "ticker",
                    "feature_date",
                    forward_return_column,
                    negative_return_column,
                ]
            ]
        )

    return pd.concat(target_frames, ignore_index=True)


def attach_targets_to_features(
    feature_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    horizon_days: int = 20,
) -> pd.DataFrame:
    """Left-join targets onto a feature frame using asset/date keys."""

    if feature_frame.empty:
        return feature_frame.copy()

    forward_return_column = f"target_forward_return_{horizon_days}d"
    negative_return_column = f"target_negative_return_{horizon_days}d"
    return feature_frame.merge(
        target_frame[
            [
                "asset_id",
                "feature_date",
                forward_return_column,
                negative_return_column,
            ]
        ],
        on=["asset_id", "feature_date"],
        how="left",
    )
