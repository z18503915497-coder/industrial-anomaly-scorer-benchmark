"""SP: Semantic Projector — learns to project CLIP text embeddings 
from Z_CLIP to Z_AD (anomaly-detection-optimized subspace).

Architecture: 512 → hidden → 512 MLP with bottleneck.
Category-agnostic: trained on auxiliary categories, generalizes to unseen.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SemanticProjector(nn.Module):
    """Calibrates CLIP text embeddings for industrial anomaly detection.

    f_phi(t) = W2 · ReLU(W1 · t + b1) + b2
    参数量: ~262K (hidden=256)
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.W1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.W2 = nn.Linear(hidden_dim, input_dim, bias=True)

        self._init_weights()

    def _init_weights(self):
        """Near-identity initialization: SP starts close to identity mapping."""
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.zeros_(self.W1.bias)
        nn.init.xavier_uniform_(self.W2.weight)
        nn.init.zeros_(self.W2.bias)

    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        """Project text embeddings.

        Args:
            text_features: (..., D) — CLIP text embeddings (L2-normalized or raw)

        Returns:
            calibrated: (..., D) — calibrated text embeddings
        """
        h = F.relu(self.W1(text_features))
        calibrated = self.W2(h)
        # Re-normalize to unit sphere
        calibrated = F.normalize(calibrated, dim=-1)
        return calibrated

    def calibrate(self, text_features: torch.Tensor) -> torch.Tensor:
        """Alias for forward, returns normalized result."""
        return self.forward(text_features)


def contrastive_alignment_loss(
    calibrated_normal: torch.Tensor,
    visual_normal_features: torch.Tensor,
    visual_defect_features: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Contrastive loss for SP training.

    Pulls calibrated normal text toward visual normal patches,
    pushes away from visual defect patches.

    Args:
        calibrated_normal: (D,) — SP(normal_text_center)
        visual_normal_features: (N_n, D) — normal patch features
        visual_defect_features: (N_d, D) — defect patch features
        temperature: softmax temperature

    Returns:
        loss: scalar
    """
    # Positive: normal text → normal visual patches
    pos_sim = F.cosine_similarity(
        calibrated_normal.unsqueeze(0), visual_normal_features
    )  # (N_n,)

    # Negative: normal text → defect visual patches
    neg_sim = F.cosine_similarity(
        calibrated_normal.unsqueeze(0), visual_defect_features
    )  # (N_d,)

    # InfoNCE-style loss
    pos_exp = torch.exp(pos_sim / temperature).mean()  # average across patches
    neg_exp = torch.exp(neg_sim / temperature).mean()

    loss = -torch.log(pos_exp / (pos_exp + neg_exp + 1e-8))
    return loss
