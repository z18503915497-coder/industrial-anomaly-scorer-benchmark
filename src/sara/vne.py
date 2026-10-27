"""VNE: Visual Normalcy Encoder — GMM-based density estimation.

Alternatives to memory banks: parametric distribution modeling.
Convergence: O(N^{-1/2}) vs O(N^{-2/D}) for k-NN at D=512.
"""
import numpy as np
from typing import Optional, Tuple
from sklearn.mixture import GaussianMixture


class VisualNormalcyEncoder:
    """Category-agnostic normality distribution estimator.

    训练: 在 auxiliary categories 上 fit 一个共享 GMM.
    适配: 在 target category 上用 few-shot normal patches 更新 GMM 参数.
    推理: 对每个 patch 返回 negative log-likelihood 作为 anomaly score.
    """

    def __init__(self, n_components: int = 5, covariance_type: str = "full"):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.gmm: Optional[GaussianMixture] = None
        self._fitted = False

    # ================================================================
    # Training: fit on auxiliary categories
    # ================================================================

    def fit(self, features: np.ndarray) -> None:
        """用 normal patch features 拟合 GMM.

        Args:
            features: (N, D) — auxiliary categories 的 normal patches
        """
        if features.shape[0] < self.n_components * 10:
            raise ValueError(
                f"样本数 ({features.shape[0]}) 不足: 需要 ≥ {self.n_components * 10}"
            )

        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            max_iter=200,
            n_init=3,
            random_state=42,
            verbose=0,
        )
        self.gmm.fit(features)
        self._fitted = True

    # ================================================================
    # Adaptation: few-shot normal images for target category
    # ================================================================

    def adapt(self, features: np.ndarray) -> None:
        """用 target category 的 normal patches 快速适配 GMM 参数.

        仅更新 GMM 的 means (保持 weights 和 covariances 不变),
        因为 covariance structure 在 auxiliary categories 上已学好。

        Args:
            features: (M, D) — target category 的 normal patches
        """
        if not self._fitted:
            raise RuntimeError("VNE 尚未 fit — 请先调用 fit()")

        # 用 target features 重新估计 GMM (warm-start from current params)
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            max_iter=50,
            n_init=1,
            random_state=42,
            verbose=0,
            means_init=self.gmm.means_,
            precisions_init=self.gmm.precisions_,
            weights_init=self.gmm.weights_,
        )
        self.gmm.fit(features)

    # ================================================================
    # Inference: anomaly scoring
    # ================================================================

    def score_samples(self, features: np.ndarray) -> np.ndarray:
        """返回每个 patch 的 anomaly score.

        score = -log p(v | normal)

        Args:
            features: (N, D)

        Returns:
            scores: (N,) — higher = more anomalous
        """
        if not self._fitted:
            raise RuntimeError("VNE 尚未 fit")
        log_likelihood = self.gmm.score_samples(features)  # (N,)
        return -log_likelihood  # negative log-likelihood

    def get_params(self) -> dict:
        """返回 GMM 参数供调试."""
        if not self._fitted:
            return {}
        return {
            "n_components": self.n_components,
            "weights": self.gmm.weights_.tolist(),
            "means_norm": [float(np.linalg.norm(m)) for m in self.gmm.means_],
        }

    def is_fitted(self) -> bool:
        return self._fitted


# ================================================================
# Utility: 从 patch grid 中提取 normal patches
# ================================================================

def extract_normal_patches(
    patch_features: np.ndarray,
    mask: Optional[np.ndarray] = None,
    purity_threshold: float = 0.05,
) -> np.ndarray:
    """从 patch grid 中提取 normal/background patches.

    Args:
        patch_features: (h, w, D) — L2-normalized CLIP patch features
        mask: (H, W) — GT anomaly mask, 0=background (optional)
        purity_threshold: 背景 patch 的最大 defect pixel ratio

    Returns:
        patches: (N, D) — background patch features
    """
    h, w, D = patch_features.shape

    if mask is not None:
        # 需要把 mask resize 到 patch grid
        from PIL import Image
        mask_resized = np.array(
            Image.fromarray(mask.astype(np.uint8)).resize((w, h), Image.NEAREST)
        )
        bg_mask = mask_resized <= (purity_threshold * 255)
        patches = patch_features[bg_mask]
    else:
        patches = patch_features.reshape(-1, D)

    return patches
