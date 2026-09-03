import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from milearn.network.module.base import BaseNetwork
from milearn.network.module.hopt import StepwiseHopt, DEFAULT_PARAM_GRID
from milearn.network.module.utils import silence_and_seed_lightning
from numpy import ndarray
from typing import Any
from pytorch_lightning.trainer.states import TrainerFn
from torch.utils.data.dataloader import DataLoader
from torch import Tensor
from typing import List
from typing import Union


class DataModule(pl.LightningDataModule):
    """Lightning data module for plain (non-bagged) instance-level or pooled-bag datasets."""

    def __init__(self, x: ndarray, y: Any = None, batch_size: int = 128, num_workers: int = 0, val_split: float = 0.2) -> None:
        """Store the data, labels, and loader settings for later dataset construction in setup()."""
        super().__init__()
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

    def setup(self, stage: TrainerFn = None) -> None:
        """Build tensors and split into train/validation datasets when labels are given."""
        x_tensor = torch.tensor(self.x, dtype=torch.float32)
        if self.y is not None:
            y_tensor = torch.tensor(self.y, dtype=torch.float32).view(-1, 1)
            dataset = TensorDataset(x_tensor, y_tensor)
            n_val = int(len(dataset) * self.val_split)
            seed = torch.Generator().manual_seed(42)
            self.train_ds, self.val_ds = random_split(dataset, [len(dataset) - n_val, n_val], generator=seed)
        else:
            self.dataset = TensorDataset(x_tensor)

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader; raises if no labels were provided."""
        if self.y is None:
            raise ValueError("No labels provided, cannot create train loader")
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader; raises if no labels were provided."""
        if self.y is None:
            raise ValueError("No labels provided, cannot create val loader")
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self) -> DataLoader:
        """Return the prediction dataloader over all samples."""
        dataset = self.dataset
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)


class MLPNetwork(BaseNetwork):
    """Plain multi-layer perceptron for instance-level prediction."""

    def __init__(self, **kwargs: Any) -> None:
        """Forward arguments to BaseNetwork and re-seed Lightning."""
        super().__init__(**kwargs)
        silence_and_seed_lightning(seed=self.hparams.random_seed)

    def forward(self, X: Tensor) -> Tensor:
        """Transform and score a batch of instances."""
        H = self.instance_transformer(X)
        y_score = self.bag_estimator(H)
        y_pred = self.prediction(y_score)
        return y_pred

    def training_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        """Compute and log the training loss for one batch."""
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        """Compute and log the validation loss for one batch."""
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def predict_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        """Run a forward pass for one prediction batch."""
        x = batch[0]
        return self.forward(x)

    def fit(self, x: ndarray, y: Union[List[float], List[int], ndarray]) -> Any:
        """Build the network layers, prepare the datamodule, and train the model."""
        self._create_basic_layers(input_layer_size=x[0].shape[-1], hidden_layer_sizes=self.hparams.hidden_layer_sizes)
        datamodule = DataModule(
            x, y, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, val_split=0.2
        )
        self._create_and_fit_trainer(datamodule)
        return self

    def predict(self, x: ndarray) -> ndarray:
        """Return predictions for the given instances."""
        datamodule = DataModule(x, y=None, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
        outputs = self._trainer.predict(self, datamodule=datamodule)
        y_pred = torch.cat(outputs, dim=0).cpu().numpy().flatten()
        return y_pred


class BagWrapperMLPNetwork(MLPNetwork, StepwiseHopt):
    """MLP applied to a pooled bag representation, so the whole bag is scored as one instance."""

    def __init__(self, pool: str = "mean", **kwargs: Any) -> None:
        """Store the pooling strategy and forward remaining arguments to MLPNetwork."""
        super().__init__(**kwargs)
        self.pool = pool
        self.save_hyperparameters()

    def hopt(self, x, y, param_grid=None, verbose=True):
        """Run stepwise hyperparameter optimization, restricting the grid to pooling methods this class supports."""
        if param_grid is None:
            param_grid = DEFAULT_PARAM_GRID
        param_grid = dict(param_grid)
        valid_pools = ["mean", "sum", "max", "lse"]
        if param_grid.get("pool"):
            param_grid["pool"] = [i for i in param_grid["pool"] if i in valid_pools]
        return super().hopt(x, y, param_grid, verbose=verbose)

    def _pool_bags(self, X: List[ndarray]) -> ndarray:
        """Pool each bag's instances into a single vector using the configured strategy."""
        if self.pool == "mean":
            return np.asarray([np.mean(bag, axis=0) for bag in X])
        elif self.pool == "sum":
            return np.asarray([np.sum(bag, axis=0) for bag in X])
        elif self.pool == "max":
            return np.asarray([np.max(bag, axis=0) for bag in X])
        elif self.pool == "lse":
            return np.asarray([np.log(np.sum(np.exp(bag), axis=0)) for bag in X])
        else:
            raise RuntimeError(f"Unknown pooling strategy: {self.pool}")

    def fit(self, X: List[ndarray], Y: Union[List[float], List[int]]):
        """Pool each bag to a single vector, then fit the underlying MLP."""
        X = self._pool_bags(X)
        return super().fit(X, Y)

    def predict(self, X: List[ndarray]) -> ndarray:
        """Pool each bag to a single vector, then predict with the underlying MLP."""
        X = self._pool_bags(X)
        return super().predict(X)


class InstanceWrapperMLPNetwork(MLPNetwork, StepwiseHopt):
    """MLP applied per instance, with bag-level predictions obtained by pooling instance predictions."""

    def __init__(self, pool: str = "mean", **kwargs: Any) -> None:
        """Store the pooling strategy, forward remaining arguments to MLPNetwork, and validate the pool."""
        super().__init__(**kwargs)
        self.pool = pool
        self.save_hyperparameters()
        if self.pool != "mean":
            raise ValueError(f"Pooling strategy '{self.pool}' is not recognized.")

    def hopt(self, x, y, param_grid=None, verbose=True):
        """Run stepwise hyperparameter optimization, restricting the grid to pooling methods this class supports."""
        if param_grid is None:
            param_grid = DEFAULT_PARAM_GRID
        param_grid = dict(param_grid)
        valid_pools = ["mean"]
        if param_grid.get("pool"):
            param_grid["pool"] = [i for i in param_grid["pool"] if i in valid_pools]
        return super().hopt(x, y, param_grid, verbose=verbose)

    def fit(self, X: List[ndarray], Y: Union[List[float], List[int]]):
        """Flatten bags into individual instances, each labeled with its bag's label, then fit the MLP."""
        Y = np.hstack([np.full(len(bag), lb) for bag, lb in zip(X, Y)])
        X = np.vstack(np.asarray(X, dtype=object)).astype(np.float32)
        return super().fit(X, Y)

    def predict(self, bags: List[ndarray]) -> ndarray:
        """Predict per instance, then average the predictions within each bag."""
        y_pred = []
        for bag in bags:
            bag = bag.reshape(-1, bag.shape[-1])
            inst_pred = super().predict(bag)
            bag_pred = np.mean(inst_pred, axis=0)
            y_pred.append(bag_pred)
        y_pred = np.asarray(y_pred)
        return y_pred
