from .module.attention import AdditiveAttentionNetwork, HopfieldAttentionNetwork, SelfAttentionNetwork
from .module.base import BaseClassifier
from .module.classic import BagNetwork, InstanceNetwork
from .module.dynamic import DynamicPoolingNetwork, MarginLoss
from .module.mlp import BagWrapperMLPNetwork, InstanceWrapperMLPNetwork
from typing import Any
from torch import Tensor


class BagNetworkClassifier(BagNetwork, BaseClassifier):
    """Bag-level network with mean/sum/max/lse pooling for classification."""

    def __init__(self, pool: str = "mean", **kwargs: Any) -> None:
        """Forward the pooling strategy and remaining arguments to BagNetwork."""
        super().__init__(pool=pool, **kwargs)


class InstanceNetworkClassifier(InstanceNetwork, BaseClassifier):
    """Instance-level network with per-instance predictions pooled to bag-level for classification."""

    def __init__(self, pool: str = "mean", **kwargs: Any) -> None:
        """Forward the pooling strategy and remaining arguments to InstanceNetwork."""
        super().__init__(pool=pool, **kwargs)


class AdditiveAttentionNetworkClassifier(AdditiveAttentionNetwork, BaseClassifier):
    """Additive attention network adapted for classification."""

    def __init__(self, **kwargs: Any) -> None:
        """Forward all arguments to AdditiveAttentionNetwork."""
        super().__init__(**kwargs)


class SelfAttentionNetworkClassifier(SelfAttentionNetwork, BaseClassifier):
    """Self-attention network adapted for classification."""

    def __init__(self, **kwargs: Any) -> None:
        """Forward all arguments to SelfAttentionNetwork."""
        super().__init__(**kwargs)


class HopfieldAttentionNetworkClassifier(HopfieldAttentionNetwork, BaseClassifier):
    """Hopfield-style attention network adapted for classification."""

    def __init__(self, **kwargs: Any) -> None:
        """Forward all arguments to HopfieldAttentionNetwork."""
        super().__init__(**kwargs)


class BagWrapperMLPNetworkClassifier(BagWrapperMLPNetwork, BaseClassifier):
    """MLP network with bag-level pooling for classification."""

    def __init__(self, **kwargs: Any) -> None:
        """Forward all arguments to BagWrapperMLPNetwork."""
        super().__init__(**kwargs)


class InstanceWrapperMLPNetworkClassifier(InstanceWrapperMLPNetwork, BaseClassifier):
    """MLP network with instance-level predictions pooled to bag-level for classification."""

    def __init__(self, **kwargs: Any) -> None:
        """Forward all arguments to InstanceWrapperMLPNetwork."""
        super().__init__(**kwargs)


class DynamicPoolingNetworkClassifier(DynamicPoolingNetwork, BaseClassifier):
    """Dynamic pooling network adapted for classification."""

    def __init__(self, **kwargs: Any) -> None:
        """Forward all arguments to DynamicPoolingNetwork."""
        super().__init__(**kwargs)

    def loss(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Compute margin-based loss between predicted bag norms and true labels."""
        margin_loss = MarginLoss()
        loss = margin_loss(y_pred, y_true)
        return loss
