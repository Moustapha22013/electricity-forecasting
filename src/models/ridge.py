"""Ridge regression on the causal features, with a fixed (data-independent) design expansion.

Why an expansion: a linear model on `hour` cannot draw a two-peak daily profile, and it
cannot learn that `lag_1` matters at 1 h but not at 8 h. We therefore add, when the needed
columns are present:
* one-hot(hour)                      -> any daily shape
* one-hot(hour) x is_weekend         -> a different shape on weekends
* one-hot(hour) x lag_1              -> a horizon-specific weight for the last known value
The expansion is a pure function of each row (no fitted parameter), so it cannot leak.
StandardScaler (fitted on the training rows only, inside the pipeline) then Ridge (L2).

    min_beta ||y - X beta||^2 + alpha ||beta||^2   ->   beta = (X'X + alpha I)^-1 X'y
The L2 penalty shrinks correlated coefficients together (lag/rolling features correlate up
to 0.83, phase 4) instead of letting them explode with opposite signs.
"""
from __future__ import annotations

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .feature_model import FeatureForecaster


def expand_design(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    if "hour" in X:
        H = pd.get_dummies(X["hour"].astype(int)).reindex(columns=range(24), fill_value=0).astype(float)
        H.columns = [f"h{h}" for h in range(24)]
        out = out.join(H)
        if "is_weekend" in X:
            out = out.join(H.mul(X["is_weekend"], axis=0).add_prefix("we_"))
        if "lag_1" in X:
            out = out.join(H.mul(X["lag_1"], axis=0).add_prefix("lag1_"))
    return out


def make_ridge(features: list[str], alpha: float = 10.0, pi_alpha: float = 0.2, **kw) -> FeatureForecaster:
    """`alpha` = L2 penalty of Ridge; `pi_alpha` = miscoverage of the prediction interval (0.2 -> 80 %)."""
    est = make_pipeline(FunctionTransformer(expand_design), StandardScaler(), Ridge(alpha=alpha))
    return FeatureForecaster(f"ridge_a{alpha:g}", est, features, alpha=pi_alpha, **kw)
