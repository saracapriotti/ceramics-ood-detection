import os
import sys

import torch
import torch.nn as nn

import config

ARCH = config.ARCH


# -- Multi-view wrapper ---------------------------------------------------------

class MultiCropWrapper(nn.Module):
    """Transparent wrapper around the model (DynamicViT, ResNet or timm ViT).

    - Input [B, 3, H, W] (training): forwarded to the model unchanged, output
      returned as is.
    - Input [B, K, 3, H, W] (evaluation with K views per image): views are
      processed in chunks of EVAL_CHUNK_SIZE, logits are averaged over the K
      views and [B, C] logits are returned.

    OOD features: the output of model.norm is averaged over tokens and then
    over views, and passed through self.norm (Identity). Hooks registered on
    `model.norm` (see ood_utils) therefore receive one [B, D] feature per
    IMAGE, even in multi-view mode (D = 512 for DynamicViT, 2048 for ResNet50,
    384 for DeiT-small).

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


# -- Comparison architectures ---------------------------------------------------
# ResNet and ViT do not have the final LayerNorm of the D-ViT, so each one is
# wrapped in a module exposing a `.norm` (Identity) placed exactly where the
# features that go to the classifier are: after global average pooling for the
# ResNet, after token pooling for the ViT. The extracted feature is therefore
# always "the input of the classification head", whatever the network. The head
# is a single Linear in all architectures, so only the backbone changes.

class ResNetBackbone(nn.Module):
    """torchvision ResNet with a .norm where the features are.

    `children()[:-1]` keeps everything up to and including global average
    pooling and drops only the final fully connected layer. The feature size
    (512 for ResNet18, 2048 for ResNet50) is read from m.fc.in_features."""

    def __init__(self, num_classes: int, name: str = "resnet50"):
        super().__init__()
        from torchvision import models

        constructor = getattr(models, name, None)
        if constructor is None:
            raise ValueError(f"torchvision has no model called {name!r}")
        m = constructor(weights="IMAGENET1K_V1")
        dim = m.fc.in_features
        self.features = nn.Sequential(*list(m.children())[:-1])
        self.norm = nn.Identity()          # feature extraction point
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)    # [B, D]
        x = self.norm(x)
        return self.head(x)


class ViTBackbone(nn.Module):
    """timm ViT with a .norm where the pooled features are.

    num_classes=0 makes timm return the already aggregated features instead of
    the logits, so we do not depend on the internal module names of timm."""

    def __init__(self, num_classes: int, name: str = None):
        super().__init__()
        import timm

        name = name or config.TIMM_MODEL
        self.vit = timm.create_model(name, pretrained=True, num_classes=0)
        self.norm = nn.Identity()          # feature extraction point
        self.head = nn.Linear(self.vit.num_features, num_classes)

    def forward(self, x):
        x = self.vit(x)                    # [B, D] already pooled
        x = self.norm(x)
        return self.head(x)


# -- Model builders -------------------------------------------------------------

def _build_dvit(num_classes: int, pretrained_path: str, viz_mode: bool) -> nn.Module:
    """DynamicViT: ImageNet-pretrained backbone + new classification heads.
    The DynamicViT repo is imported here, so ResNet and ViT do not need it."""
    root = os.environ.get("DYNAMICVIT_REPO", config.DYNAMICVIT_REPO)
    if root not in sys.path:
        sys.path.insert(0, root)
    from models.dylvvit import LVViTDiffPruning  # DynamicViT/models/dylvvit.py

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
    return model


def build_model(
    num_classes: int,
    pretrained_path: str = None,
    device: torch.device | None = None,
    viz_mode: bool = True,
) -> nn.Module:
    """Builds the model chosen by config.ARCH, wrapped in a MultiCropWrapper."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if ARCH == "dvit":
        inner = _build_dvit(num_classes, pretrained_path or config.PRETRAINED_PATH, viz_mode)
    elif ARCH.startswith("resnet"):
        inner = ResNetBackbone(num_classes, name=ARCH)
    elif ARCH == "vit":
        inner = ViTBackbone(num_classes)
    else:
        raise ValueError(f"Unknown config.ARCH: {ARCH!r}. Allowed values: 'dvit', "
                         "'vit', or the name of a torchvision ResNet ('resnet50', ...).")

    wrapped = MultiCropWrapper(inner, chunk_size=config.EVAL_CHUNK_SIZE)
    wrapped.to(device)
    print(f"Model ready on {device} (architecture: {ARCH})")
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
