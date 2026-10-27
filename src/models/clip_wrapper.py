# ============================================================
# CLIP 统一接口模块
# 功能: 加载 CLIP 变体，提供统一的视觉/文本编码接口
# 支持: CPU / CUDA 自动切换
#
# encode_image 策略:
#   OpenCLIP model.visual() 默认只返回 pooled token (B,512)，
#   不返回 patch tokens。因此我们拆开三步:
#     _embeds()      → conv + cls token + pos embed + ln_pre
#     transformer()  → 全部 transformer blocks
#     _pool()        → ln_post + global_pool → (pooled, tokens)
#   然后对 pooled 和 tokens 都应用 proj 投影到 512 维。
# ============================================================

import torch
import torch.nn.functional as F
from PIL import Image
from typing import Tuple, List, Optional
import open_clip


class CLIPWrapper:
    """CLIP 模型的统一接口。

    支持的变体:
      - ViT-B-16 (openai)     ← 输入 224×224，patch 14×14，embed_dim=512
      - ViT-B-32 (openai)     ← 输入 224×224，patch 7×7，  embed_dim=512
      - ViT-L-14 (openai)     ← 输入 224×224，patch 16×16，embed_dim=768
      - ViT-L-14-336 (openai) ← 输入 336×336，patch 24×24，embed_dim=768

    Args:
        model_name: 模型名称，如 "ViT-B-16"
        pretrained: 预训练权重，如 "openai"
        device: "cpu", "cuda:0", 或 "auto"
    """

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device

        print(f"[CLIP] 加载模型: {model_name} ({pretrained}) on {device}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.to(device)
        self.model.eval()

        self.visual = self.model.visual
        self.patch_size = getattr(self.visual, "patch_size", None)
        self.image_size_val = getattr(self.visual, "image_size", None)

    # ================================================================
    # encode_image
    # ================================================================

    def encode_image(self, image: Image.Image) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """编码单张图像 → 全局视觉特征 + patch 特征。

        走 OpenCLIP 官方三步:
          _embeds()  → patch embedding + CLS token + position + ln_pre
          transformer() → 全部 transformer blocks
          _pool()    → ln_post + global_pool → (pooled, tokens)

        然后对 pooled 和 tokens 都应用 proj（OpenCLIP forward 默认只对
        pooled 做 proj），确保两者都在 512 维联合嵌入空间中，与文本特征对齐。

        Returns:
            global_feat: (1, 512)  L2 归一化的全局特征
            patch_feat:  (1, H, W, 512)  L2 归一化的 patch 特征
        """
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # ———— 步骤 1-3: OpenCLIP 官方内部 forward ————
            # _embeds:  conv1 → reshape → cat CLS token → +pos embed → ln_pre
            x = self.visual._embeds(img_tensor)              # (1, 197, 768) for ViT-B-16

            # transformer:  all blocks
            x = self.visual.transformer(x)                   # (1, 197, 768)

            # _pool:  ln_post → global_pool → (pooled, tokens)
            pooled, tokens = self.visual._pool(x)            # pooled=(1,768), tokens=(1,196,768)

            # ———— 步骤 4: 对 pooled 和 tokens 都应用 proj ————
            # OpenCLIP forward 默认只对 pooled 做 proj，不对 tokens 做。
            # 为了让 patch tokens 也进入 512 维嵌入空间（与文本特征对齐），
            # 我们对 tokens 也应用相同的 proj 矩阵。
            if self.visual.proj is not None:
                pooled = pooled @ self.visual.proj            # (1, 512)
                tokens = tokens @ self.visual.proj            # (1, 196, 512)

            # ———— 步骤 5: L2 归一化 ————
            global_feat = F.normalize(pooled, dim=-1)         # (1, 512)
            patch_feat = F.normalize(tokens, dim=-1)          # (1, 196, 512)

            # ———— 步骤 6: 重塑为空间形状 ————
            h = w = int(patch_feat.shape[1] ** 0.5)
            patch_feat = patch_feat.reshape(1, h, w, -1)     # (1, 14, 14, 512)

        return global_feat, patch_feat

    # ================================================================
    # encode_text
    # ================================================================

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """编码文本列表 → 文本特征。

        Returns:
            (N, 512) L2 归一化的文本特征
        """
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            text_feat = self.model.encode_text(tokens)
            text_feat = F.normalize(text_feat, dim=-1)
        return text_feat

    # ================================================================
    # compute_similarity
    # ================================================================

    def compute_similarity(self, image: Image.Image, texts: List[str]) -> torch.Tensor:
        """计算图像与文本的相似度（用于 ISD 计算）。"""
        img_feat, _ = self.encode_image(image)
        text_feat = self.encode_text(texts)
        sim = (img_feat @ text_feat.T).squeeze(0)
        return sim

    # ================================================================
    # 工具方法
    # ================================================================

    def get_embed_dim(self) -> int:
        return self.model.visual.output_dim if hasattr(self.model.visual, "output_dim") else 512

    def get_image_size(self) -> Tuple[int, int]:
        if self.image_size_val:
            if isinstance(self.image_size_val, int):
                return (self.image_size_val, self.image_size_val)
            return self.image_size_val
        return (224, 224)

