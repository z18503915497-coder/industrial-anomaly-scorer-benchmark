"""Pixel Refinement — lightweight edge-aware score smoothing.

Uses patch-level cosine affinity to propagate anomaly scores
across visually similar neighbors. No parameters, no training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PixelRefinement(nn.Module):
    """Edge-aware anomaly score refinement via local affinity.

    For each patch (i,j), compute cosine similarity with its 8 spatial
    neighbors in the patch feature space. Softmax the similarities to
    produce attention weights, then compute the weighted average of
    neighboring anomaly scores.

    Effect: smooths anomaly scores within visually homogeneous regions
    while preserving boundaries between dissimilar patches.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        score_map: torch.Tensor,
        patch_feature: torch.Tensor,
    ) -> torch.Tensor:
        """Refine anomaly score map using local patch affinity.

        Args:
            score_map: (H, W) — raw anomaly scores
            patch_feature: (H, W, D) — L2-normalized patch features

        Returns:
            refined: (H, W) — smoothed anomaly scores, same shape
        """
        H, W = score_map.shape
        D = patch_feature.shape[-1]

        # Flatten to (N, D) and (N,)
        features = patch_feature.reshape(-1, D)   # (H*W, D)
        scores = score_map.reshape(-1)             # (H*W,)

        N = H * W

        # Build neighbor indices for each patch (8-connectivity)
        # Use a padded grid to handle boundaries cleanly
        pad_f = F.pad(
            features.reshape(1, H, W, D).permute(0, 3, 1, 2),
            (1, 1, 1, 1), mode='replicate'
        )  # (1, D, H+2, W+2)
        pad_f = pad_f.squeeze(0).permute(1, 2, 0)  # (H+2, W+2, D)

        pad_s = F.pad(
            scores.reshape(1, 1, H, W),
            (1, 1, 1, 1), mode='replicate'
        ).squeeze()  # (H+2, W+2)

        # 8-neighbor offsets (excluding center)
        offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

        refined = torch.zeros(N, device=scores.device, dtype=scores.dtype)

        for di, dj in offsets:
            # Neighbor features at offset
            ni = 1 + di  # row index in padded grid
            nj = 1 + dj  # col index
            nbr_f = pad_f[ni:ni+H, nj:nj+W, :].reshape(N, D)  # (N, D)
            nbr_s = pad_s[ni:ni+H, nj:nj+W].reshape(N)          # (N,)

            # Cosine similarity with neighbor
            sim = F.cosine_similarity(features, nbr_f, dim=1)  # (N,)

            # Softmax across all 8 neighbors
            # We accumulate weighted scores; weights = softmax(sim) per patch
            # Use exp(sim) for simplicity (cos_sim in [-1,1] is fine)
            weight = torch.exp(sim)  # (N,)
            refined += weight * nbr_s

        # Normalize by sum of weights
        # Recompute unnormalized sum (same loop but simpler)
        weight_sum = torch.zeros(N, device=scores.device)
        for di, dj in offsets:
            ni, nj = 1 + di, 1 + dj
            nbr_f = pad_f[ni:ni+H, nj:nj+W, :].reshape(N, D)
            sim = F.cosine_similarity(features, nbr_f, dim=1)
            weight_sum += torch.exp(sim)

        refined = refined / (weight_sum + 1e-8)

        # Blend: 0.3 * original + 0.7 * refined (keep some original signal)
        alpha = 0.7
        refined = alpha * refined + (1 - alpha) * scores

        return refined.reshape(H, W)


def refine_score_map(
    score_map: np.ndarray,
    patch_feature: np.ndarray,
) -> np.ndarray:
    """NumPy wrapper for PixelRefinement (no torch dependency).

    Args:
        score_map: (H, W) np.float32
        patch_feature: (H, W, D) np.float32 — L2-normalized

    Returns:
        refined: (H, W) np.float32
    """
    score_t = torch.from_numpy(score_map.astype(np.float32))
    patch_t = torch.from_numpy(patch_feature.astype(np.float32))
    refiner = PixelRefinement()
    with torch.no_grad():
        refined_t = refiner(score_t, patch_t)
    return refined_t.detach().cpu().numpy()
