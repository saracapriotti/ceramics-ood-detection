import os
import sys

import torch
import torch.nn as nn

import config

# -- DynamicViT repo imports ----------------------------------------------------
_DYNVIT_ROOT = os.environ.get("DYNAMICVIT_REPO", config.DYNAMICVIT_REPO)
if _DYNVIT_ROOT not in sys.path:
    sys.path.insert(0, _DYNVIT_ROOT)

from models.dylvvit import LVViTDiffPruning  # DynamicViT/models/dylvvit.py


# -- Multi-view wrapper ---------------------------------------------------------

class MultiCropWrapper(nn.Module):
    """Transparent wrapper around DynamicViT.

    - Input [B, 3, H, W] (training): forwarded to the model unchanged, output
      returned as is.
    - Input [B, K, 3, H, W] (evaluation with K views per image): views are
      processed in chunks of EVAL_CHUNK_SIZE, logits are averaged over the K
      views and [B, C] logits are returned.

    OOD features: the output of model.norm is averaged over tokens and then
    over views, and passed through self.norm (Identity). Hooks registered on
    `model.norm` (see ood_utils) therefore receive one [B, 512] feature per
    IMAGE, even in multi-view mode.

    state_dict()/load_state_dict() are delegated to the inner model, so
    checkpoints keep the original keys. Attributes not found on the wrapper
    (head, aux_head, ...) are read from the inner model.
    """

    def __init__(self, model: nn.Module, chunk_size: int = 128):
        super().__init__()
        self.model = model
        self.norm = nn.Identity()
        self.chunk_size = chunk_size
        self._feat_buffer = []
        self._hook = model.norm.register_forward_hook(self._capture_features)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            modules = self.__dict__.get("_modules", {})
            if "model" in modules:
                return getattr(modules["model"], name)
            raise

    def __setattr__(self, name, value):
        # `distill` switches DynamicViT to returning the pruning decisions
        # needed by the keep-ratio loss: it must reach the inner model.
        if name == "distill" and "_modules" in self.__dict__ and "model" in self._modules:
            setattr(self._modules["model"], name, value)
            return
        super().__setattr__(name, value)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True, **kwargs):
        return self.model.load_state_dict(state_dict, strict=strict)

    def _capture_features(self, module, inp, out):
        pooled = out.mean(dim=1) if out.dim() == 3 else out
        self._feat_buffer.append(pooled.detach().float())

    def forward(self, x: torch.Tensor):
        self._feat_buffer.clear()

        if x.dim() == 4:
            out = self.model(x)
            if self._feat_buffer:
                self.norm(torch.cat(self._feat_buffer, dim=0))
            self._feat_buffer.clear()
            return out

        if x.dim() != 5:
            raise ValueError(f"Expected input [B,3,H,W] or [B,K,3,H,W], got {tuple(x.shape)}")
        if self.training:
            raise RuntimeError("Multi-view input [B,K,3,H,W] is only supported in eval mode.")

        B, K = x.shape[:2]
        flat = x.flatten(0, 1)
        logits_parts = []
        for start in range(0, flat.shape[0], self.chunk_size):
            out = self.model(flat[start:start + self.chunk_size])
            logits = out[0] if isinstance(out, (tuple, list)) else out
            logits_parts.append(logits.float())

        logits = torch.cat(logits_parts, dim=0).view(B, K, -1).mean(dim=1)
        feats = torch.cat(self._feat_buffer, dim=0).view(B, K, -1).mean(dim=1)
        self._feat_buffer.clear()
        self.norm(feats)
        return logits


# -- Model builders -------------------------------------------------------------

def build_model(
    num_classes: int,
    pretrained_path: str = config.PRETRAINED_PATH,
    device: torch.device | None = None,
    viz_mode: bool = True,
) -> nn.Module:
    """DynamicViT: ImageNet-pretrained backbone + new classification heads."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LVViTDiffPruning(
        patch_size=config.PATCH_SIZE,
        embed_dim=config.EMBED_DIM,
        depth=config.DEPTH,
        num_heads=config.NUM_HEADS,
        mlp_ratio=config.MLP_RATIO,
        p_emb=config.P_EMB,
        skip_lam=config.SKIP_LAM,
        return_dense=config.RETURN_DENSE,
        mix_token=config.MIX_TOKEN,
        pruning_loc=config.PRUNING_LOC,
        token_ratio=config.TOKEN_RATIO,
        viz_mode=viz_mode,
    )

    checkpoint = torch.load(pretrained_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    model.head = nn.Linear(model.head.in_features, num_classes)
    model.aux_head = nn.Linear(model.aux_head.in_features, num_classes)

    wrapped = MultiCropWrapper(model, chunk_size=config.EVAL_CHUNK_SIZE)
    wrapped.to(device)
    print(f"Model ready on {device}")
    return wrapped


def load_model(
    checkpoint_path: str,
    num_classes: int,
    device: torch.device | None = None,
) -> nn.Module:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_classes=num_classes, device=device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    if "model" in state_dict:
        state_dict = state_dict["model"]

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Loaded model weights from {checkpoint_path}")
    return model
