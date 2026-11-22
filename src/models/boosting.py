"""Histogram gradient boosting on the causal features.

F_M(x) = sum_m nu * f_m(x), each tree f_m fitted to the negative gradient of the loss.
* loss="absolute_error": gradient = sign(y - F) -> the model targets the conditional MEDIAN,
  the minimiser of the MAE we report; robust to the heavy right tail (skew +1.27).
* loss="squared_error": targets the conditional mean.
Trees capture interactions (hour x weekday x lag_1, holiday x hour) without hand-made terms.

`early_stopping=False`: sklearn's early stopping holds out a RANDOM (shuffled) fraction of
the training rows, which mixes time periods; the number of iterations is a tuned
hyper-parameter instead.
"""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor

from .feature_model import FeatureForecaster


def make_boosting(features: list[str], loss: str = "absolute_error", learning_rate: float = 0.05,
                  max_iter: int = 300, max_leaf_nodes: int = 31, min_samples_leaf: int = 40,
                  l2_regularization: float = 1.0, seed: int = 0, **kw) -> FeatureForecaster:
    est = HistGradientBoostingRegressor(loss=loss, learning_rate=learning_rate, max_iter=max_iter,
                                        max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                                        l2_regularization=l2_regularization, early_stopping=False,
                                        random_state=seed)
    name = f"gbm_{loss.split('_')[0]}_lr{learning_rate:g}_it{max_iter}_lf{max_leaf_nodes}"
    return FeatureForecaster(name, est, features, **kw)
