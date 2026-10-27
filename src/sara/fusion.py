"""AF: Adaptive Fusion — uncertainty-weighted combination of visual and text scores.

When V-score is highly confident (very high or very low), it dominates.
When V-score is near the decision boundary, text provides semantic anchoring.
"""
import numpy as np


def adaptive_fusion(
    visual_score: np.ndarray,
    text_score: np.ndarray,
    theta: float = 0.5,
    gamma: float = 0.1,
) -> np.ndarray:
    """Uncertainty-weighted fusion of visual and text anomaly scores.

    alpha(p) = sigmoid((S_v(p) - theta) / gamma)
    S(p) = alpha(p) * S_v(p) + (1 - alpha(p)) * S_t(p)

    Args:
        visual_score: (N,) — VNE anomaly scores (negative log-likelihood)
        text_score:   (N,) — calibrated WinCLIP anomaly scores [0,1]
        theta: decision boundary offset
        gamma: sharpness of transition

    Returns:
        fused_score: (N,) — combined anomaly score
    """
    # Sigmoid confidence with numerical stability
    x = np.clip((visual_score - theta) / gamma, -50.0, 50.0)
    alpha = 1.0 / (1.0 + np.exp(-x))

    # Both scores must be [0,1] for meaningful fusion
    # V-score: normalize to [0,1] via per-image min-max
    v_min, v_max = visual_score.min(), visual_score.max()
    v_range = v_max - v_min
    if v_range > 1e-6:
        visual_score_norm = (visual_score - v_min) / v_range
    else:
        visual_score_norm = np.zeros_like(visual_score)

    fused = alpha * visual_score_norm + (1.0 - alpha) * text_score
    return fused.astype(np.float32)
