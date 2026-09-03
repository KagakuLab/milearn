import torch
from torch import nn

from .base import BaseNetwork, instance_dropout
from typing import Any
from torch import Tensor
from typing import Tuple


class BaseAttentionNetwork(BaseNetwork):
    def __init__(self, tau: float = 1.0, **kwargs: Any) -> None:
        """Store the attention temperature and forward remaining arguments to BaseNetwork."""
        super().__init__(**kwargs)
        self.tau = tau

    def _create_special_layers(self, input_layer_size: int, hidden_layer_sizes: tuple[int, ...]):
        """Create the attention layers for this network."""
        self._create_attention(hidden_layer_sizes)

    def _create_attention(self, hidden_layer_sizes):
        """Define the attention mechanism; must be implemented in subclasses."""
        raise NotImplementedError

    def forward(self, bags: Tensor, inst_mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Transform instances, compute attention-weighted bag embeddings, then score the bag."""
        # 1. Compute instance embeddings
        inst_embed = self.instance_transformer(bags)

        # 2. Apply instance dropout
        inst_mask = instance_dropout(inst_mask, self.hparams.instance_dropout, self.training)
        inst_embed = inst_mask * inst_embed

        # 3. Compute instance attention weights
        bag_embed, inst_weights = self.compute_attention(inst_embed, inst_mask)

        # 4. Compute final bag prediction
        bag_score = self.bag_estimator(bag_embed)
        bag_pred = self.prediction(bag_score)

        return bag_embed, inst_weights, bag_pred

    def compute_attention(self, H, M):
        """Compute attention weights and the resulting bag embedding; must be implemented in subclasses."""
        raise NotImplementedError


class AdditiveAttentionNetwork(BaseAttentionNetwork):
    def _create_attention(self, hidden_layer_sizes: Tuple[int, int, int]) -> None:
        """Create the additive attention MLP."""
        self.attention = nn.Sequential(
            nn.Linear(hidden_layer_sizes[-1], hidden_layer_sizes[-1]), nn.Tanh(), nn.Linear(hidden_layer_sizes[-1], 1)
        )

    def compute_attention(self, inst_embed: Tensor, inst_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Score each instance, mask padding, softmax to weights, and combine into a bag embedding."""
        # 1. Compute logits
        inst_logits = self.attention(inst_embed) / self.tau

        # 2. Mask padded instances
        mask_bool = inst_mask.squeeze(-1).bool()
        inst_logits = inst_logits.masked_fill(~mask_bool.unsqueeze(-1), float("-inf"))

        # 3. Compute weights
        inst_weights = torch.softmax(inst_logits, dim=1)

        # 4. Weighted sum to get bag embedding
        bag_embed = torch.sum(inst_weights * inst_embed, dim=1, keepdim=True)

        return bag_embed, inst_weights


class SelfAttentionNetwork(BaseAttentionNetwork):
    def _create_attention(self, hidden_layer_sizes: Tuple[int, int, int]) -> None:
        """Create the query, key, and value projections for self-attention."""
        D = hidden_layer_sizes[-1]
        self.q_proj = nn.Linear(D, D)
        self.k_proj = nn.Linear(D, D)
        self.v_proj = nn.Linear(D, D)

    def compute_attention(self, inst_embed: Tensor, inst_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute scaled dot-product self-attention over instances and pool into a bag embedding."""
        # 1. Project to Q, K, V
        Q = self.q_proj(inst_embed)
        K = self.k_proj(inst_embed)
        V = self.v_proj(inst_embed)

        # 2. Compute scaled dot-product attention
        inst_logits = torch.matmul(Q, K.transpose(1, 2)) / (self.tau * (inst_embed.shape[-1] ** 0.5))

        # 3. Mask invalid instances
        mask_bool = inst_mask.squeeze(-1).bool()
        inst_logits = inst_logits.masked_fill(~mask_bool.unsqueeze(1), float("-inf"))

        # 4. Compute attention weights
        inst_weights = torch.softmax(inst_logits, dim=-1)  # (B, N, N)

        # 5. Reduce to per-instance / Incoming (who gets attended to)
        inst_weights = inst_weights.mean(dim=1, keepdim=True).transpose(1, 2)  # (B, N, 1)

        # 6. Weighted sum of values -> bag embedding
        bag_embed = torch.sum(inst_weights * V, dim=1, keepdim=True)  # (B, 1, D)

        return bag_embed, inst_weights


class HopfieldAttentionNetwork(BaseAttentionNetwork):
    def __init__(self, tau: float = 1.0, **kwargs: Any) -> None:
        """Store the attention scale (beta) and forward remaining arguments to BaseNetwork."""
        super().__init__(**kwargs)
        self.beta = tau

    def _create_attention(self, hidden_layer_sizes: Tuple[int, int, int]) -> None:
        """Create the learnable query vector used for Hopfield-style attention."""
        self.query_vector = nn.Parameter(torch.randn(1, hidden_layer_sizes[-1]))

    def compute_attention(self, inst_embed: Tensor, inst_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Attend from a single learned query vector to all instances and pool into a bag embedding."""
        B, N, D = inst_embed.shape

        # 1. Expand query vector to batch
        q = self.query_vector.unsqueeze(0).expand(B, -1, -1)  # [B, 1, D]

        # 2. Compute scores
        inst_logits = self.beta * torch.bmm(q, inst_embed.transpose(1, 2))  # [B, 1, N]

        # 3. Mask invalid instances
        mask_bool = inst_mask.squeeze(-1).bool()
        inst_logits = inst_logits.masked_fill(~mask_bool.unsqueeze(1), float("-inf"))

        # 4. Attention weights
        inst_weights = torch.softmax(inst_logits, dim=-1)
        inst_weights = inst_weights.transpose(1, 2)

        # 5. Compute bag embedding
        bag_embed = torch.bmm(inst_weights.transpose(1, 2), inst_embed)  # [B, 1, D]

        return bag_embed, inst_weights
