"""
Small, dependency-light model implementations for forecasting baselines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def _to_numpy(features: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Convert feature input into a clean float matrix."""

    if isinstance(features, pd.DataFrame):
        matrix = features.to_numpy(dtype=float)
    else:
        matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Model features must be a 2D matrix.")
    return matrix


def _coerce_target(target: pd.Series | np.ndarray) -> np.ndarray:
    """Convert model targets into a 1D float array."""

    return np.asarray(target, dtype=float).reshape(-1)


class RidgeRegressor:
    """Closed-form ridge regression baseline for forward-return estimation."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.feature_mean_: np.ndarray | None = None
        self.feature_scale_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def fit(self, features: pd.DataFrame | np.ndarray, target: pd.Series | np.ndarray) -> "RidgeRegressor":
        x = _to_numpy(features)
        y = _coerce_target(target)
        self.feature_mean_ = x.mean(axis=0)
        self.feature_scale_ = x.std(axis=0)
        self.feature_scale_[self.feature_scale_ == 0.0] = 1.0
        x_scaled = (x - self.feature_mean_) / self.feature_scale_

        design = np.column_stack([np.ones(len(x_scaled)), x_scaled])
        penalty = np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        solution = np.linalg.pinv(design.T @ design + self.alpha * penalty) @ design.T @ y

        self.intercept_ = float(solution[0])
        self.coef_ = solution[1:]
        return self

    def predict(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.feature_mean_ is None or self.feature_scale_ is None:
            raise ValueError("The ridge model must be fit before prediction.")
        x = _to_numpy(features)
        x_scaled = (x - self.feature_mean_) / self.feature_scale_
        return self.intercept_ + (x_scaled @ self.coef_)


@dataclass
class _TreeNode:
    is_leaf: bool
    value: float
    probability: float | None = None
    feature_index: int | None = None
    threshold: float | None = None
    left: "_TreeNode | None" = None
    right: "_TreeNode | None" = None


class _DecisionTree:
    """Simple CART-style decision tree used inside the local random forests."""

    def __init__(
        self,
        task: str,
        max_depth: int = 4,
        min_samples_leaf: int = 8,
        max_features: int | None = None,
        random_state: int | None = None,
    ) -> None:
        self.task = task
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = np.random.default_rng(random_state)
        self.root_: _TreeNode | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_DecisionTree":
        self.root_ = self._build_tree(x, y, depth=0)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise ValueError("The tree must be fit before prediction.")
        return np.asarray([self._predict_row(row, self.root_) for row in x], dtype=float)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("Probability output is only supported for classification trees.")
        if self.root_ is None:
            raise ValueError("The tree must be fit before prediction.")
        return np.asarray([self._predict_probability(row, self.root_) for row in x], dtype=float)

    def _leaf_node(self, y: np.ndarray) -> _TreeNode:
        if self.task == "regression":
            return _TreeNode(is_leaf=True, value=float(np.mean(y)))
        probability = float(np.mean(y))
        return _TreeNode(is_leaf=True, value=float(probability >= 0.5), probability=probability)

    def _impurity(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        if self.task == "regression":
            return float(np.var(y) * len(y))
        positive_rate = np.mean(y)
        gini = 1.0 - (positive_rate ** 2 + (1.0 - positive_rate) ** 2)
        return float(gini * len(y))

    def _build_tree(self, x: np.ndarray, y: np.ndarray, depth: int) -> _TreeNode:
        if depth >= self.max_depth or len(y) < self.min_samples_leaf * 2 or np.allclose(y, y[0]):
            return self._leaf_node(y)

        split = self._find_best_split(x, y)
        if split is None:
            return self._leaf_node(y)

        feature_index, threshold = split
        left_mask = x[:, feature_index] <= threshold
        right_mask = ~left_mask

        return _TreeNode(
            is_leaf=False,
            value=float(np.mean(y)),
            feature_index=feature_index,
            threshold=threshold,
            left=self._build_tree(x[left_mask], y[left_mask], depth + 1),
            right=self._build_tree(x[right_mask], y[right_mask], depth + 1),
        )

    def _find_best_split(self, x: np.ndarray, y: np.ndarray) -> tuple[int, float] | None:
        n_samples, n_features = x.shape
        feature_count = self.max_features or max(1, int(np.sqrt(n_features)))
        candidate_features = self.random_state.choice(
            n_features,
            size=min(feature_count, n_features),
            replace=False,
        )

        best_score = np.inf
        best_split: tuple[int, float] | None = None
        base_score = self._impurity(y)

        for feature_index in candidate_features:
            column = x[:, feature_index]
            finite_values = column[np.isfinite(column)]
            if len(finite_values) < self.min_samples_leaf * 2:
                continue

            thresholds = np.unique(np.quantile(finite_values, q=np.linspace(0.1, 0.9, 9)))
            for threshold in thresholds:
                left_mask = column <= threshold
                right_mask = ~left_mask
                if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                    continue

                score = self._impurity(y[left_mask]) + self._impurity(y[right_mask])
                if score < best_score and score < base_score:
                    best_score = score
                    best_split = (int(feature_index), float(threshold))

        return best_split

    def _predict_row(self, row: np.ndarray, node: _TreeNode) -> float:
        if node.is_leaf:
            return node.value
        if row[node.feature_index] <= node.threshold:
            return self._predict_row(row, node.left)
        return self._predict_row(row, node.right)

    def _predict_probability(self, row: np.ndarray, node: _TreeNode) -> float:
        if node.is_leaf:
            return float(node.probability or 0.0)
        if row[node.feature_index] <= node.threshold:
            return self._predict_probability(row, node.left)
        return self._predict_probability(row, node.right)


class RandomForestRegressor:
    """Bootstrap-aggregated regression trees for nonlinear return estimation."""

    def __init__(
        self,
        n_estimators: int = 40,
        max_depth: int = 4,
        min_samples_leaf: int = 8,
        max_features: int | None = None,
        random_state: int | None = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees_: list[_DecisionTree] = []

    def fit(self, features: pd.DataFrame | np.ndarray, target: pd.Series | np.ndarray) -> "RandomForestRegressor":
        x = _to_numpy(features)
        y = _coerce_target(target)
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []

        for estimator_index in range(self.n_estimators):
            sample_index = rng.integers(0, len(x), size=len(x))
            tree = _DecisionTree(
                task="regression",
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=int(rng.integers(0, 1_000_000) + estimator_index),
            )
            tree.fit(x[sample_index], y[sample_index])
            self.trees_.append(tree)
        return self

    def predict(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.trees_:
            raise ValueError("The random forest must be fit before prediction.")
        x = _to_numpy(features)
        predictions = np.vstack([tree.predict(x) for tree in self.trees_])
        return predictions.mean(axis=0)


class RandomForestClassifier:
    """Bootstrap-aggregated classification trees for downside-risk probabilities."""

    def __init__(
        self,
        n_estimators: int = 40,
        max_depth: int = 4,
        min_samples_leaf: int = 8,
        max_features: int | None = None,
        random_state: int | None = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees_: list[_DecisionTree] = []

    def fit(self, features: pd.DataFrame | np.ndarray, target: pd.Series | np.ndarray) -> "RandomForestClassifier":
        x = _to_numpy(features)
        y = _coerce_target(target)
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []

        for estimator_index in range(self.n_estimators):
            sample_index = rng.integers(0, len(x), size=len(x))
            tree = _DecisionTree(
                task="classification",
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=int(rng.integers(0, 1_000_000) + estimator_index),
            )
            tree.fit(x[sample_index], y[sample_index])
            self.trees_.append(tree)
        return self

    def predict_proba(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.trees_:
            raise ValueError("The random forest must be fit before prediction.")
        x = _to_numpy(features)
        probabilities = np.vstack([tree.predict_proba(x) for tree in self.trees_]).mean(axis=0)
        return np.column_stack([1.0 - probabilities, probabilities])

    def predict(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(features)[:, 1]
        return (probabilities >= 0.5).astype(int)


def get_model_factories() -> dict[str, dict[str, Any]]:
    """Return factories for the supported regression and classification models."""

    return {
        "regression": {
            "ridge_regression": lambda: RidgeRegressor(alpha=2.0),
            "random_forest_regressor": lambda: RandomForestRegressor(
                n_estimators=48,
                max_depth=5,
                min_samples_leaf=6,
                random_state=42,
            ),
        },
        "classification": {
            "random_forest_classifier": lambda: RandomForestClassifier(
                n_estimators=48,
                max_depth=5,
                min_samples_leaf=6,
                random_state=42,
            ),
        },
    }
