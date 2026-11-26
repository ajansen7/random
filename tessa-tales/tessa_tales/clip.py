"""CLIP on the Mac: turns pictures and little descriptions into comparable numbers."""

import warnings

import numpy as np


class Clip:
    def __init__(self, model_name: str, pretrained: str):
        warnings.filterwarnings("ignore")
        import open_clip
        import torch

        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = model.eval().to(self.device)
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.tag = f"{model_name}-{pretrained}"

    def _norm(self, feats) -> np.ndarray:
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.float().cpu().numpy()

    def embed_images(self, images, batch: int = 32) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(images), batch):
                x = self.torch.stack([self.preprocess(im) for im in images[i:i + batch]]).to(self.device)
                out.append(self._norm(self.model.encode_image(x)))
        return np.concatenate(out) if out else np.zeros((0, 1), np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        with self.torch.no_grad():
            return self._norm(self.model.encode_text(self.tokenizer(texts).to(self.device)))
