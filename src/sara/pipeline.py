"""SARAPipeline: Full Semantic Alignment via Reciprocal Adaptation.

整合 VNE + SP + AF, 提供统一的接口供 experiment 调用.
"""
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image

from src.sara.vne import VisualNormalcyEncoder, extract_normal_patches
from src.sara.sp import SemanticProjector
from src.sara.fusion import adaptive_fusion


class SARAPipeline:
    """SARA: Semantic Alignment via Reciprocal Adaptation.

    Usage:
        sara = SARAPipeline(clip_model, temperature=0.07)
        sara.train_sp(auxiliary_normal_patches, auxiliary_defect_patches, text_prompts)
        sara.build_vne(normal_reference_patches)
        anomaly_map = sara.predict(image, text_prompts)
    """

    def __init__(
        self,
        clip_model,
        n_gmm_components: int = 5,
        sp_hidden_dim: int = 256,
        temperature: float = 0.07,
        af_theta: float = 0.5,
        af_gamma: float = 0.1,
        use_refinement: bool = False,
        device: str = "cuda:0",
    ):
        self.clip_model = clip_model
        self.temperature = temperature
        self.af_theta = af_theta
        self.af_gamma = af_gamma
        self.use_refinement = use_refinement
        self.device = device

        self.vne = VisualNormalcyEncoder(n_components=n_gmm_components)
        self.sp = SemanticProjector(input_dim=512, hidden_dim=sp_hidden_dim).to(device)
        self.sp_optimizer = torch.optim.AdamW(self.sp.parameters(), lr=1e-4, weight_decay=1e-4)

        self._sp_trained = False
        self._vne_fitted = False

    # ================================================================
    # SP Training
    # ================================================================

    def train_sp(
        self,
        normal_patches: np.ndarray,
        normal_text: List[str],
        anomaly_text: List[str],
        epochs: int = 50,
        batch_size: int = 128,
        margin: float = 0.2,
        sep_lambda: float = 0.5,
        verbose: bool = True,
    ) -> dict:
        """在 auxiliary categories 上训练 Semantic Projector (normal-only).

        Strict zero-shot: SP 仅使用 auxiliary normal patches + text descriptions.
        不使用 defect masks 或 target category 的任何数据。

        Loss:
          L_align = 1 - mean(cos(cal_norm_text, normal_patch))
          L_sep   = relu(margin - (cos(cal_norm_text, patch) - cos(cal_anom_text, patch)))
          L_total = L_align + λ * L_sep

        Args:
            normal_patches: (N_n, 512) — auxiliary normal patch features
            normal_text: 文本 prompt list
            anomaly_text: 文本 prompt list
            epochs: 训练轮数
            batch_size: patch batch size
            margin: separation margin
            sep_lambda: loss weight for separation term
            verbose: 打印训练日志

        Returns:
            dict with keys: losses, raw_text_dist, cal_text_dist, improvement
        """
        # Pre-encode raw text centers (frozen, no grad)
        with torch.no_grad():
            n_feat = self.clip_model.encode_text(normal_text)      # (N_t, 512)
            a_feat = self.clip_model.encode_text(anomaly_text)     # (N_t, 512)
            raw_n_center = F.normalize(n_feat.mean(dim=0, keepdim=True), dim=-1)  # (1, 512)
            raw_a_center = F.normalize(a_feat.mean(dim=0, keepdim=True), dim=-1)  # (1, 512)
            raw_text_dist = 1.0 - float(F.cosine_similarity(raw_n_center, raw_a_center))

        # PyTorch tensors (anomaly text also on device for L_sep)
        normal_t = torch.tensor(normal_patches, dtype=torch.float32, device=self.device)
        raw_n = raw_n_center.to(self.device)
        raw_a = raw_a_center.to(self.device)

        self.sp.train()
        history = {"losses": [], "align_losses": [], "sep_losses": [],
                    "cal_text_dists": []}

        for epoch in range(epochs):
            epoch_align = 0.0
            epoch_sep = 0.0
            n_batches = max(1, len(normal_patches) // batch_size)

            for _ in range(n_batches):
                n_idx = np.random.choice(len(normal_patches),
                                         min(batch_size, len(normal_patches)),
                                         replace=False)
                n_batch = normal_t[n_idx]  # (B, 512)

                # SP only calibrates normal text; anomaly text stays frozen
                cal_n = self.sp(raw_n).squeeze(0)  # (512,)

                # L_align: maximize cos(cal_norm_text, normal_patch)
                cos_n = F.cosine_similarity(cal_n.unsqueeze(0), n_batch)  # (B,)
                loss_align = (1.0 - cos_n).mean()

                # L_sep: normal patch closer to cal_normal than raw_anomaly (frozen anchor)
                raw_a_batch = raw_a.expand(n_batch.shape[0], -1)  # (B, 512)
                cos_a = F.cosine_similarity(n_batch, raw_a_batch)  # (B,)
                margin_loss = F.relu(margin - (cos_n - cos_a))
                loss_sep = margin_loss.mean()

                loss = loss_align + sep_lambda * loss_sep

                self.sp_optimizer.zero_grad()
                loss.backward()
                self.sp_optimizer.step()

                epoch_align += loss_align.item()
                epoch_sep += loss_sep.item()

            avg_align = epoch_align / max(1, n_batches)
            avg_sep = epoch_sep / max(1, n_batches)
            avg_total = avg_align + sep_lambda * avg_sep

            history["losses"].append(avg_total)
            history["align_losses"].append(avg_align)
            history["sep_losses"].append(avg_sep)

            # Compute calibrated text distance (SP(normal) vs raw_anomaly, no grad)
            with torch.no_grad():
                cal_dist = 1.0 - float(F.cosine_similarity(
                    F.normalize(cal_n.unsqueeze(0), dim=-1),
                    raw_a,  # frozen, not through SP
                ))
            history["cal_text_dists"].append(cal_dist)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  SP epoch {epoch+1:3d}/{epochs}  "
                      f"align={avg_align:.6f}  sep={avg_sep:.6f}  "
                      f"total={avg_total:.6f}  cal_dist={cal_dist:.6f}")

        self._sp_trained = True
        self.sp.eval()

        # Final text separation: fresh forward pass in eval mode
        with torch.no_grad():
            final_cal_n = F.normalize(self.sp(raw_n), dim=-1)
            final_cal_dist = 1.0 - float(F.cosine_similarity(final_cal_n, raw_a))

        history["raw_text_dist"] = raw_text_dist
        history["cal_text_dist"] = final_cal_dist

        if verbose:
            print(f"\n  SP Training Complete")
            print(f"    Raw text separation:       {raw_text_dist:.6f}")
            print(f"    Calibrated text separation: {final_cal_dist:.6f}")

        return history

    # ================================================================
    # VNE Building
    # ================================================================

    def build_vne(self, normal_reference_patches: np.ndarray) -> None:
        """用 target category 的 normal reference patches 构建 VNE.

        Args:
            normal_reference_patches: (M, 512) — target category normal patches
        """
        if normal_reference_patches.shape[0] < self.vne.n_components * 10:
            raise ValueError(
                f"Normal reference patches ({normal_reference_patches.shape[0]}) 不足. "
                f"需要 ≥ {self.vne.n_components * 10}"
            )
        self.vne.fit(normal_reference_patches)
        self._vne_fitted = True

    # ================================================================
    # Prediction
    # ================================================================

    def predict(
        self,
        image: Image.Image,
        normal_text: List[str],
        anomaly_text: List[str],
        image_size: Tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        """对单张图像生成 SARA anomaly map.

        Args:
            image: PIL Image (RGB)
            normal_text: normal prompt list
            anomaly_text: anomaly prompt list
            image_size: output map size

        Returns:
            anomaly_map: (H, W) np.float32, range [0, 1]
        """
        if not self._vne_fitted:
            raise RuntimeError("VNE 尚未构建 — 请先调用 build_vne()")
        if not self._sp_trained:
            print("[SARA] WARNING: SP 未训练 — 使用 raw CLIP text embeddings")

        # 1. Extract patch features
        _, patch_feat = self.clip_model.encode_image(image)
        patch_np = patch_feat.detach().cpu().numpy()[0]  # (h, w, D)
        h, w, D = patch_np.shape
        patch_flat = patch_np.reshape(-1, D)

        # 2. Text encoding + SP calibration
        n_feat = self.clip_model.encode_text(normal_text)     # (N_n, D)
        a_feat = self.clip_model.encode_text(anomaly_text)    # (N_a, D)
        n_center = F.normalize(n_feat.mean(dim=0, keepdim=True), dim=-1)
        a_center = F.normalize(a_feat.mean(dim=0, keepdim=True), dim=-1)

        if self._sp_trained:
            n_center = self.sp(n_center.to(self.device))       # (1, D)
            a_center = self.sp(a_center.to(self.device))

        n_center_np = n_center.detach().cpu().numpy()
        a_center_np = a_center.detach().cpu().numpy()

        # 3. Text anomaly score (calibrated WinCLIP)
        sim_n = np.dot(patch_flat, n_center_np.T).squeeze(-1)
        sim_a = np.dot(patch_flat, a_center_np.T).squeeze(-1)
        delta = sim_a - sim_n
        z = np.clip(delta / self.temperature, -50.0, 50.0)
        text_score = 1.0 / (1.0 + np.exp(-z))  # sigmoid, (h*w,)

        # 4. Visual anomaly score (VNE)
        visual_score = self.vne.score_samples(patch_flat)  # (h*w,)

        # 5. Adaptive fusion
        score = adaptive_fusion(
            visual_score, text_score,
            theta=self.af_theta, gamma=self.af_gamma,
        )

        # 5b. Pixel Refinement (optional, edge-aware smoothing)
        if self.use_refinement:
            from src.sara.refinement import refine_score_map
            score_2d = score.reshape(h, w).astype(np.float32)
            score_2d = refine_score_map(score_2d, patch_np)
            score = score_2d.reshape(-1)

        # 6. Reshape + upsample
        anomaly_map = score.reshape(h, w).astype(np.float32)
        anomaly_map_t = torch.from_numpy(anomaly_map).unsqueeze(0).unsqueeze(0)
        am = F.interpolate(
            anomaly_map_t, size=image_size, mode="bilinear", align_corners=False
        ).squeeze().detach().cpu().numpy().astype(np.float32)

        return am

    # ================================================================
    # Diagnostics
    # ================================================================

    def diagnose_text_separation(self, normal_text: List[str], anomaly_text: List[str]) -> dict:
        """诊断校准前后的 text separation."""
        with torch.no_grad():
            n_feat = self.clip_model.encode_text(normal_text)
            a_feat = self.clip_model.encode_text(anomaly_text)
            n_raw = F.normalize(n_feat.mean(dim=0, keepdim=True), dim=-1)
            a_raw = F.normalize(a_feat.mean(dim=0, keepdim=True), dim=-1)
            raw_dist = 1.0 - float(F.cosine_similarity(n_raw, a_raw))

            if self._sp_trained:
                n_cal = self.sp(n_raw.to(self.device))
                a_cal = self.sp(a_raw.to(self.device))
                cal_dist = 1.0 - float(F.cosine_similarity(n_cal, a_cal))
            else:
                cal_dist = raw_dist

        return {
            "raw_cos_distance": raw_dist,
            "calibrated_cos_distance": cal_dist,
            "improvement": cal_dist / (raw_dist + 1e-8),
            "sp_trained": self._sp_trained,
        }

    def get_vne_params(self) -> dict:
        return self.vne.get_params()

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.sp.state_dict(), path / "sp.pt")
        # VNE (sklearn GMM): 用 joblib
        import joblib
        if self._vne_fitted:
            joblib.dump(self.vne.gmm, path / "vne_gmm.joblib")

    def load(self, path: Path) -> None:
        self.sp.load_state_dict(torch.load(path / "sp.pt", map_location=self.device))
        self._sp_trained = True
        import joblib
        gmm_path = path / "vne_gmm.joblib"
        if gmm_path.exists():
            self.vne.gmm = joblib.load(gmm_path)
            self.vne._fitted = True
            self._vne_fitted = True
