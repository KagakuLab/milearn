import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from numpy import ndarray
from typing import Any
from typing import List
from typing import Optional

class BagScaler(BaseEstimator, TransformerMixin):
    """Wraps a scikit-learn scaler, fitting it on all instances across bags and applying it per bag."""
    def __init__(self, scaler: MinMaxScaler = None) -> None:
        """Store the underlying scaler, defaulting to MinMaxScaler() if none is given."""
        self.scaler = scaler if scaler is not None else MinMaxScaler()

    def fit(self, x: List[ndarray], y: Optional[Any] = None) -> "BagScaler":
        """Fit the scaler on all instances pooled from every bag."""
        all_instances = np.vstack(x)  # stack all bags for fitting
        self.scaler.fit(all_instances, y)
        return self

    def transform(self, x):
        """Apply the fitted scaler to each bag independently."""
        x_scaled = [self.scaler.transform(bag) for bag in x]
        return x_scaled

    def fit_transform(self, X, y=None, **fit_params):
        """Fit the scaler and transform the bags in a single step."""
        return self.fit(X, y).transform(X)


class BagMinMaxScaler(BagScaler):
    """BagScaler using sklearn's MinMaxScaler."""
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(scaler=MinMaxScaler(**kwargs))


class BagStandardScaler(BagScaler):
    """BagScaler using sklearn's StandardScaler."""
    def __init__(self, **kwargs):
        super().__init__(scaler=StandardScaler(**kwargs))


class BagMaxAbsScaler(BagScaler):
    """BagScaler using sklearn's MaxAbsScaler."""
    def __init__(self, **kwargs):
        super().__init__(scaler=MaxAbsScaler(**kwargs))


class BagRobustScaler(BagScaler):
    """BagScaler using sklearn's RobustScaler."""
    def __init__(self, **kwargs):
        super().__init__(scaler=RobustScaler(**kwargs))
