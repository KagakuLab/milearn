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
    """PyTorch Lightning data module for handling bags and labels."""

    def __init__(self, x: ndarray, y: Any = None, batch_size: int = 128, num_workers: int = 0, val_split: float = 0.2) -> None:
        """Initialize DataModule.

        Args:
            x (list | np.ndarray): list of bags (instances).
            y (list | np.ndarray, optional): bag-level labels.
            batch_size (int): batch size for dataloaders.
            num_workers (int): number of workers for dataloaders.
            val_split (float): fraction of data used for validation.
        """
        super().__init__()
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

    def setup(self, stage: TrainerFn = None) -> None:
        """Prepare datasets for training, validation, or prediction.

        Args:
            stage (str, optional): stage name ('fit', 'predict', etc.).
        """
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
        """Training dataloader.

        Returns:
            DataLoader: PyTorch DataLoader for training.
        """
        if self.y is None:
            raise ValueError("No labels provided, cannot create train loader")
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        """Validation dataloader.

        Returns:
            DataLoader: PyTorch DataLoader for validation.
        """
        if self.y is None:
            raise ValueError("No labels provided, cannot create val loader")
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self) -> DataLoader:
        """Prediction dataloader.

        Returns:
            DataLoader: PyTorch DataLoader for prediction.
        """
        dataset = self.dataset
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)


class MLPNetwork(BaseNetwork):
    """Multi-layer perceptron network for bag-level or instance-level
    prediction."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize MLPNetwork.

        Args:
            **kwargs: additional arguments for BaseNetwork.
        """
        super().__init__(**kwargs)
        silence_and_seed_lightning(seed=self.hparams.random_seed)

    def forward(self, X: Tensor) -> Tensor:
        """Forward pass.

        Args:
            X (torch.Tensor): input instances.

        Returns:
            torch.Tensor: predictions.
        """
        H = self.instance_transformer(X)
        y_score = self.bag_estimator(H)
        y_pred = self.prediction(y_score)
        return y_pred

    def training_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        """Training step.

        Args:
            batch (tuple): (x, y) batch data.
            batch_idx (int): index of batch.

        Returns:
            torch.Tensor: computed loss.
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        """Validation step.

        Args:
            batch (tuple): (x, y) batch data.
            batch_idx (int): index of batch.

        Returns:
            torch.Tensor: computed loss.
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def predict_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        """Prediction step.

        Args:
            batch (tuple): input batch.
            batch_idx (int): index of batch.

        Returns:
            torch.Tensor: predictions.
        """
        x = batch[0]
        return self.forward(x)

    def fit(self, x: ndarray, y: Union[List[float], List[int], ndarray]) -> Any:
        """Fit the model on the provided dataset.

        Args:
            x (list | np.ndarray): input bags.
            y (list | np.ndarray): labels.

        Returns:
            MLPNetwork: trained model instance.
        """
        self._create_basic_layers(input_layer_size=x[0].shape[-1], hidden_layer_sizes=self.hparams.hidden_layer_sizes)
        datamodule = DataModule(
            x, y, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, val_split=0.2
        )
        self._create_and_fit_trainer(datamodule)
        return self

    def predict(self, x: ndarray) -> ndarray:
        """Predict on new data.

        Args:
            x (list | np.ndarray): input bags.

        Returns:
            np.ndarray: predicted values.
        """
        datamodule = DataModule(x, y=None, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
        outputs = self._trainer.predict(self, datamodule=datamodule)
        y_pred = torch.cat(outputs, dim=0).cpu().numpy().flatten()
        return y_pred


class BagWrapperMLPNetwork(MLPNetwork, StepwiseHopt):
    """Wrapper for MLPNetwork to handle bag-level pooling."""

    def __init__(self, pool: str = "mean", **kwargs: Any) -> None:
        """Initialize BagWrapperMLPNetwork.

        Args:
            pool (str): pooling strategy, currently only "mean" is supported.
            **kwargs: additional arguments for MLPNetwork.
        """
        super().__init__(**kwargs)
        self.pool = pool
        self.save_hyperparameters()

    def hopt(self, x, y, param_grid=DEFAULT_PARAM_GRID, verbose=True):
        """Hyperparameter optimization with support for pooling methods."""
        valid_pools = ["mean", "sum", "max", "lse"]
        if param_grid.get("pool"):
            param_grid["pool"] = [i for i in param_grid["pool"] if i in valid_pools]
        return super().hopt(x, y, param_grid, verbose=verbose)

    def fit(self, X: List[ndarray], Y: Union[List[float], List[int]]):
        """Fit the model after pooling bag representations."""

        if self.pool == "mean":
            X = np.asarray([np.mean(bag, axis=0) for bag in X])
        elif self.pool == "sum":
            X = np.asarray([np.sum(bag, axis=0) for bag in X])
        elif self.pool == "max":
            X = np.asarray([np.max(bag, axis=0) for bag in X])
        elif self.pool == "lse":
            X = np.asarray([np.log(np.sum(np.exp(bag), axis=0)) for bag in X])
        else:
            raise RuntimeError(f"Unknown pooling strategy: {self.pool}")
        return super().fit(X, Y)

    def predict(self, X: List[ndarray]) -> ndarray:
        """Predict after pooling bag representations.

        Args:
            X (list | np.ndarray): input bags.

        Returns:
            np.ndarray: predictions.
        """
        if self.pool == "mean":
            X = np.asarray([np.mean(bag, axis=0) for bag in X])
        else:
            raise RuntimeError("Unknown pooling strategy.")
        return super().predict(X)


class InstanceWrapperMLPNetwork(MLPNetwork, StepwiseHopt):
    """Wrapper for MLPNetwork to handle instance-level predictions with
    pooling."""

    def __init__(self, pool: str = "mean", **kwargs: Any) -> None:
        """Initialize InstanceWrapperMLPNetwork.

        Args:
            pool (str): pooling strategy, currently only "mean" is supported.
            **kwargs: additional arguments for MLPNetwork.
        """
        super().__init__(**kwargs)
        self.pool = pool
        self.save_hyperparameters()
        if self.pool != "mean":
            raise ValueError(f"Pooling strategy '{self.pool}' is not recognized.")

    def hopt(self, x, y, param_grid=DEFAULT_PARAM_GRID, verbose=True):
        """Hyperparameter optimization with support for pooling methods."""
        valid_pools = ["mean"]
        if param_grid.get("pool"):
            param_grid["pool"] = [i for i in param_grid["pool"] if i in valid_pools]
        return super().hopt(x, y, param_grid, verbose=verbose)

    def fit(self, X: List[ndarray], Y: Union[List[float], List[int]]):
        """Fit model after converting bags to single-instance dataset.

        Args:
            X (list | np.ndarray): input bags.
            Y (list | np.ndarray): bag labels.

        Returns:
            InstanceWrapperMLPNetwork: trained model instance.
        """
        Y = np.hstack([np.full(len(bag), lb) for bag, lb in zip(X, Y)])
        X = np.vstack(np.asarray(X, dtype=object)).astype(np.float32)
        return super().fit(X, Y)

    def predict(self, bags: List[ndarray]) -> ndarray:
        """Predict on instance-level and pool results to bag-level.

        Args:
            bags (list | np.ndarray): input bags.

        Returns:
            np.ndarray: bag-level predictions.
        """
        y_pred = []
        for bag in bags:
            bag = bag.reshape(-1, bag.shape[-1])
            inst_pred = super().predict(bag)
            bag_pred = np.mean(inst_pred, axis=0)
            y_pred.append(bag_pred)
        y_pred = np.asarray(y_pred)
        return y_pred