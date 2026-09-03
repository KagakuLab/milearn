from .base import BaseNetwork, instance_dropout
from .hopt import StepwiseHopt, DEFAULT_PARAM_GRID
from typing import Any
from torch import Tensor
from typing import Tuple


class BagNetwork(BaseNetwork, StepwiseHopt):
    """MIL network that pools instance embeddings into a bag embedding before scoring."""

    def __init__(self, pool: str = "mean", **kwargs: Any) -> None:
        """Store the pooling strategy and forward remaining arguments to BaseNetwork."""
        super().__init__(**kwargs)
        self.pool = pool

    def _pooling(self, bags: Tensor, inst_mask: Tensor) -> Tensor:
        """Aggregate instance embeddings into a bag embedding using the configured pooling."""
        if self.pool == "mean":
            bag_embed = bags.sum(axis=1) / inst_mask.sum(axis=1)
        elif self.pool == "sum":
            bag_embed = bags.sum(axis=1)
        elif self.pool == "max":
            bag_embed = bags.max(dim=1)[0]
        elif self.pool == "lse":
            bag_embed = bags.exp().sum(dim=1).log()
        else:
            raise TypeError(f"Pooling type {self.pool} is not supported.")

        bag_embed = bag_embed.unsqueeze(1)
        return bag_embed

    def forward(self, bags: Tensor, inst_mask: Tensor) -> Tuple[Tensor, None, Tensor]:
        """Transform, pool, and score a batch of bags."""
        inst_embed = self.instance_transformer(bags)
        inst_mask = instance_dropout(inst_mask, self.hparams.instance_dropout, self.training)
        inst_embed = inst_mask * inst_embed
        bag_embed = self._pooling(inst_embed, inst_mask)
        bag_score = self.bag_estimator(bag_embed)
        bag_pred = self.prediction(bag_score)

        return bag_embed, None, bag_pred

    def hopt(self, x, y, param_grid=None, verbose=False):
        """Run stepwise hyperparameter optimization, restricting the grid to pooling methods this class supports."""
        if param_grid is None:
            param_grid = DEFAULT_PARAM_GRID
        param_grid = dict(param_grid)
        valid_pools = ["mean", "sum", "max", "lse"]
        if param_grid.get("pool"):
            param_grid["pool"] = [i for i in param_grid["pool"] if i in valid_pools]
        return super().hopt(x, y, param_grid, verbose=verbose)


class InstanceNetwork(BaseNetwork):
    """MIL network that scores each instance individually, then pools the predictions into a bag prediction."""

    def __init__(self, pool: str = "mean", **kwargs: Any) -> None:
        """Store the pooling strategy and forward remaining arguments to BaseNetwork."""
        super().__init__(**kwargs)
        self.pool = pool

    def _pooling(self, inst_pred: Tensor, inst_mask: Tensor) -> Tensor:
        """Aggregate instance-level predictions into a bag-level prediction using the configured pooling."""
        if self.pool == "mean":
            bag_pred = inst_pred.sum(axis=1) / inst_mask.sum(axis=1)
        elif self.pool == "sum":
            bag_pred = inst_pred.sum(axis=1)
        elif self.pool == "max":
            idx = inst_pred.abs().argmax(dim=1, keepdim=True)
            bag_pred = inst_pred.gather(1, idx).squeeze(1)
        else:
            raise TypeError(f"Pooling type {self.pool} is not supported.")
        bag_pred = bag_pred.unsqueeze(1)
        return bag_pred

    def forward(self, bags: Tensor, inst_mask: Tensor) -> Tuple[None, None, Tensor]:
        """Transform each instance, score it, then pool scores into a bag prediction."""
        inst_embed = self.instance_transformer(bags)
        inst_mask = instance_dropout(inst_mask, self.hparams.instance_dropout, self.training)
        inst_embed = inst_mask * inst_embed
        inst_score = self.bag_estimator(inst_embed)
        bag_score = self._pooling(inst_score, inst_mask)
        bag_pred = self.prediction(bag_score)

        return None, None, bag_pred

    def hopt(self, x, y, param_grid=None, verbose=True):
        """Run stepwise hyperparameter optimization, restricting the grid to pooling methods this class supports."""
        if param_grid is None:
            param_grid = DEFAULT_PARAM_GRID
        param_grid = dict(param_grid)
        valid_pools = ["mean", "sum", "max"]
        if param_grid.get("pool"):
            param_grid["pool"] = [i for i in param_grid["pool"] if i in valid_pools]
        return super().hopt(x, y, param_grid, verbose=verbose)
