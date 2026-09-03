import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseNetwork, instance_dropout
from torch import Tensor
from typing import Tuple
from typing import Any


class MarginLoss(nn.Module):
    """Margin loss for capsule-like models."""

    def __init__(self, m_pos=0.9, m_neg=0.1, alpha=0.5):
        """Initialize MarginLoss.

        Args:
            m_pos (float): positive margin threshold.
            m_neg (float): negative margin threshold.
            alpha (float): scaling factor for negative terms.
        """
        super(MarginLoss, self).__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.alpha = alpha

    def forward(self, lengths: Tensor, labels: Tensor) -> Tensor:
        """Compute margin loss.

        Args:
            lengths (torch.Tensor): predicted lengths (probabilities).
            labels (torch.Tensor): ground-truth labels.

        Returns:
            torch.Tensor: scalar margin loss.
        """
        left = F.relu(self.m_pos - lengths, inplace=True) ** 2
        right = F.relu(lengths - self.m_neg, inplace=True) ** 2
        margin_loss = labels * left + self.alpha * (1.0 - labels) * right
        return margin_loss.mean()


class Squash(nn.Module):
    """Squashing nonlinearity for capsule-like networks."""

    def forward(self, bag_embed: Tensor) -> Tensor:
        """Apply squash function.

        Args:
            bag_embed (torch.Tensor): bag embeddings.

        Returns:
            torch.Tensor: squashed embeddings.
        """
        norm = torch.norm(bag_embed, p=2, dim=2, keepdim=True)
        scale = norm**2 / (1 + norm**2) / (norm + 1e-8)
        return scale * bag_embed


class Norm(nn.Module):
    """Compute L2 norm of bag embeddings."""

    def forward(self, bag_squash: Tensor) -> Tensor:
        """Compute norm.

        Args:
            bag_squash (torch.Tensor): squashed embeddings.

        Returns:
            torch.Tensor: L2 norm values.
        """
        return torch.norm(bag_squash, p=2, dim=2, keepdim=True)


class DynamicPooling(nn.Module):
    """Dynamic routing-based pooling layer for multiple-instance learning."""

    def __init__(self, n_iter: int = 3) -> None:
        """Initialize DynamicPooling.

        Args:
            n_iter (int): number of routing iterations.
        """
        super().__init__()
        self.n_iter = n_iter

    def forward(self, inst_embed: Tensor, inst_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply dynamic pooling.

        Args:
            inst_embed (torch.Tensor): instance embeddings.
            inst_mask (torch.Tensor): mask for valid instances.

        Returns:
            tuple: (bag embeddings, instance weights).
        """
        inst_embed = inst_mask * inst_embed
        inst_logits = torch.zeros(*inst_embed.shape[:2], 1, device=inst_embed.device, dtype=inst_embed.dtype)

        for t in range(self.n_iter):
            inst_weights = torch.softmax(inst_mask * inst_logits, dim=1)
            bag_embed = torch.sum(inst_weights * inst_embed, dim=1, keepdim=True)
            bag_squash = Squash()(bag_embed)
            new_logits = torch.sum(bag_squash * inst_embed, dim=2, keepdim=True)
            inst_logits = inst_logits + new_logits

        inst_weights = torch.softmax(inst_logits, dim=1)
        return bag_squash, inst_weights


class DynamicPoolingNetwork(BaseNetwork):
    """A dynamic pooling-based multiple-instance learning network."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize DynamicPoolingNetwork.

        Args:
            **kwargs: additional arguments for BaseNetwork.
        """
        super().__init__(**kwargs)

    def _create_basic_layers(self, input_layer_size: int, hidden_layer_sizes: tuple[int, ...]):
        """Create basic layers for the network.

        Args:
            input_layer_size (int): size of input features.
            hidden_layer_sizes (tuple[int]): hidden layer sizes.
        """
        super()._create_basic_layers(input_layer_size, hidden_layer_sizes)
        self.bag_estimator = Norm()

    def _create_special_layers(self, input_layer_size: int, hidden_layer_sizes: Tuple[int, int, int]) -> None:
        """Create dynamic pooling layer.

        Args:
            input_layer_size (int): size of input features.
            hidden_layer_sizes (tuple[int]): hidden layer sizes.
        """
        self.dynamic_pooling = DynamicPooling()

    def forward(self, bags: Tensor, inst_mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass of DynamicPoolingNetwork.

        Args:
            bags (torch.Tensor): input bags of instances.
            inst_mask (torch.Tensor): instance mask.

        Returns:
            tuple: (bag embeddings, instance weights, bag predictions).
        """
        inst_embed = self.instance_transformer(bags)
        inst_mask = instance_dropout(inst_mask, self.hparams.instance_dropout, self.training)
        inst_embed = inst_mask * inst_embed
        bag_embed, inst_weights = self.dynamic_pooling(inst_embed, inst_mask)
        bag_pred = self.bag_estimator(bag_embed)

        return bag_embed, inst_weights, bag_pred
