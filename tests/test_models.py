import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from models.ds_svit import build_ds_svit
from models.baselines.registry import build_model


def test_ds_svit_forward():
    model = build_ds_svit(img_size=64, patch_size=16, embed_dim=64, depth=2, num_heads=4, fusion_heads=4)
    x = torch.randn(2, 3, 64, 64)
    prob, logit = model(x)
    assert prob.shape == (2,)
    assert logit.shape == (2,)
    assert torch.all((prob >= 0) & (prob <= 1))


def test_ds_svit_return_features():
    model = build_ds_svit(img_size=64, patch_size=16, embed_dim=64, depth=2, num_heads=4, fusion_heads=4)
    x = torch.randn(2, 3, 64, 64)
    prob, logit, feat = model(x, return_features=True)
    assert feat.shape == (2, 64)


def test_xception_forward():
    model = build_model("xception")
    x = torch.randn(2, 3, 224, 224)
    prob, logit = model(x)
    assert prob.shape == (2,)


def test_f3net_forward():
    model = build_model("f3net")
    x = torch.randn(2, 3, 224, 224)
    prob, logit = model(x)
    assert prob.shape == (2,)


def test_swin_forward():
    model = build_model("swin", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    prob, logit = model(x)
    assert prob.shape == (2,)


def test_iapl_forward():
    model = build_model("iapl", embed_dim=96)
    x = torch.randn(2, 3, 224, 224)
    prob, logit = model(x)
    assert prob.shape == (2,)


if __name__ == "__main__":
    test_ds_svit_forward()
    test_ds_svit_return_features()
    test_xception_forward()
    test_f3net_forward()
    test_swin_forward()
    test_iapl_forward()
    print("All model tests passed.")
