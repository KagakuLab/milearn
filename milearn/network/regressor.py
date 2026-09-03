import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .module.attention import AdditiveAttentionNetwork, HopfieldAttentionNetwork, SelfAttentionNetwork
from .module.base import BaseRegressor
from .module.classic import BagNetwork, InstanceNetwork
from .module.dynamic import DynamicPoolingNetwork
from .module.mlp import BagWrapperMLPNetwork, InstanceWrapperMLPNetwork
from typing import Any
from numpy import ndarray
from typing import List


class BagNetworkRegressor(BagNetwork, BaseRegressor):
    """Bag-level network with mean/sum/max/lse pooling for regression tasks."""

    def __init__(self, **kwargs: Any) -> None:
        """Forward all arguments to BagNetwork."""
        super().__init__(**kwargs)


class InstanceNetworkRegressor(InstanceNetwork, BaseRegressor):
    """Instance-level network with per-instance predictions pooled to bag-level for regression."""

    def __init__(self, **kwargs: Any) -> None:
        """Forward all arguments to InstanceNetwork."""
        super().__init__(**kwargs)


class AdditiveAttentionNetworkRegressor(AdditiveAttentionNetwork, BaseRegressor):
    """Additive attention network adapted for regression tasks."""

    def __init__(self, **kwargs: Any) -> None:
        """Forward all arguments to AdditiveAttentionNetwork."""
        super().__init__(**kwargs)


class SelfAttentionNetworkRegressor(SelfAttentionNetwork, BaseRegressor):
    """Self-attention network adapted for regression tasks."""

    def __init__(self, **kwargs: Any) -> None:
        """Forward all arguments to SelfAttentionNetwork."""
        super().__init__(**kwargs)


class HopfieldAttentionNetworkRegressor(HopfieldAttentionNetwork, BaseRegressor):
    """Hopfield-style attention network adapted for regression tasks."""

    def __init__(self, **kwargs: Any) -> None:
        """Forward all arguments to HopfieldAttentionNetwork."""
        super().__init__(**kwargs)


class BagWrapperMLPNetworkRegressor(BagWrapperMLPNetwork, BaseRegressor):
    """MLP network with bag-level pooling for regression tasks."""

    def __init__(self, **kwargs: Any) -> None:
        """Forward all arguments to BagWrapperMLPNetwork."""
        super().__init__(**kwargs)


class InstanceWrapperMLPNetworkRegressor(InstanceWrapperMLPNetwork, BaseRegressor):
    """MLP network with instance-level predictions pooled to bag-level for regression tasks."""

    def __init__(self, **kwargs: Any) -> None:
        """Forward all arguments to InstanceWrapperMLPNetwork."""
        super().__init__(**kwargs)


class DynamicPoolingNetworkRegressor(DynamicPoolingNetwork, BaseRegressor):
    """Dynamic pooling network for regression, min-max scaling targets during training and inverse-scaling predictions."""

    def __init__(self, **kwargs: Any) -> None:
        """Forward all arguments to DynamicPoolingNetwork."""
        super().__init__(**kwargs)

    def fit(self, x: List[ndarray], y: List[float]) -> "DynamicPoolingNetworkRegressor":
        """Min-max scale the targets to [0, 1], then fit the network on the scaled values."""
        y = np.array(y).reshape(-1, 1)
        self.scaler = MinMaxScaler()
        y = self.scaler.fit_transform(y).flatten()

        return super().fit(x, y)

    def predict(self, x: List[ndarray]) -> ndarray:
        """Predict scaled targets and inverse-transform them back to the original range."""
        y_pred = super().predict(x)
        y_pred = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        return y_pred
