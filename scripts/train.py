"""
Train the full TrustTour model: DS-SViT backbone wrapped in the Trust
Assessment Module, optimized with BCE on authenticity + trust-consistency
regularization (utils/losses.TrustTourLoss).

Usage:
    python scripts/train.py --config configs/trusttour.yaml --dataset cifake --root data/CIFAKE
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import build_dataset
from models.ds_svit import DSSViTConfig, build_ds_svit
from models.trust_module import TrustAssessmentModule, TrustModuleConfig
from utils.losses import TrustTourLoss
from utils.logger import CSVLogger, print_metrics
from utils.metrics import detection_metrics
from utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/trusttour.yaml")
    p.add_argument("--root", type=str, default=None, help="Override data.root from config")
    p.add_argument("--dataset", type=str, default=None, choices=["cifake", "synthbuster"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_samples_per_class", type=int, default=None,
                    help="Useful for smoke-testing on a small subset.")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_loaders(cfg: dict, args) -> tuple[DataLoader, DataLoader]:
    root = args.root or cfg["data"]["root"]
    dataset = args.dataset or cfg["data"]["dataset"]
    img_size = cfg["img_size"]
    bs = cfg["batch_size"]
    nw = cfg["data"].get("num_workers", 4)
    tourism = cfg["data"].get("tourism", True)

    train_ds = build_dataset(root, dataset, "train", img_size, tourism=tourism,
                              max_samples_per_class=args.max_samples_per_class)
    test_ds = build_dataset(root, dataset, "test", img_size, tourism=tourism,
                             max_samples_per_class=args.max_samples_per_class)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, drop_last=True)
    val_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw)
    return train_loader, val_loader


def build_trusttour_model(cfg: dict) -> TrustAssessmentModule:
    m_cfg = cfg.get("model", {})
    ds_cfg = DSSViTConfig(
        img_size=cfg["img_size"],
        embed_dim=m_cfg.get("embed_dim", 384),
        depth=m_cfg.get("depth", 6),
        num_heads=m_cfg.get("num_heads", 6),
        mlp_ratio=m_cfg.get("mlp_ratio", 4.0),
        drop_rate=m_cfg.get("drop_rate", 0.1),
        forensic_in_channels=m_cfg.get("forensic_in_channels", 64),
        fusion_heads=m_cfg.get("fusion_heads", 6),
    )
    backbone = build_ds_svit(
        img_size=ds_cfg.img_size, patch_size=ds_cfg.patch_size, embed_dim=ds_cfg.embed_dim,
        depth=ds_cfg.depth, num_heads=ds_cfg.num_heads, mlp_ratio=ds_cfg.mlp_ratio,
        drop_rate=ds_cfg.drop_rate, forensic_in_channels=ds_cfg.forensic_in_channels,
        fusion_heads=ds_cfg.fusion_heads,
    )
    t_cfg = cfg.get("trust_module", {})
    trust_cfg = TrustModuleConfig(
        feature_dim=ds_cfg.embed_dim,
        metadata_dim=t_cfg.get("metadata_dim", 8),
        source_dim=t_cfg.get("source_dim", 6),
        hidden_dim=t_cfg.get("hidden_dim", 64),
    )
    return TrustAssessmentModule(backbone, trust_cfg)


def run_epoch(model, loader, device, optimizer=None, loss_fn=None, desc="train"):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, n = 0.0, 0
    all_labels, all_probs = [], []

    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        metadata = batch["metadata"].to(device)
        source = batch["source"].to(device)

        with torch.set_grad_enabled(is_train):
            outputs = model(image, metadata, source)
            losses = loss_fn(outputs, label)
            loss = losses["loss"]

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        bs = image.shape[0]
        total_loss += loss.item() * bs
        n += bs
        all_labels.append(label.detach().cpu().numpy())
        all_probs.append(outputs["auth_prob"].detach().cpu().numpy())
        pbar.set_postfix(loss=loss.item())

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    metrics = detection_metrics(y_true, y_prob)
    metrics["loss"] = total_loss / max(n, 1)
    return metrics


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs:
        cfg["epochs"] = args.epochs
    set_seed(cfg.get("seed", 42))

    device = torch.device(args.device)
    train_loader, val_loader = build_loaders(cfg, args)

    model = build_trusttour_model(cfg).to(device)
    loss_fn = TrustTourLoss(trust_weight=cfg.get("trust_module", {}).get("trust_loss_weight", 0.3))

    opt_cfg = cfg["optimizer"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["scheduler"]["t_max"])

    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(cfg["output"]["log_dir"], filename="trusttour_train_log.csv")

    best_auc = -1.0
    patience = cfg.get("early_stopping_patience", 10)
    bad_epochs = 0

    for epoch in range(1, cfg["epochs"] + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer, loss_fn, desc=f"train e{epoch}")
        val_metrics = run_epoch(model, val_loader, device, optimizer=None, loss_fn=loss_fn, desc=f"val e{epoch}")
        scheduler.step()

        print_metrics(f"epoch {epoch} train", train_metrics)
        print_metrics(f"epoch {epoch} val", val_metrics)
        logger.log({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        if val_metrics["AUC"] > best_auc:
            best_auc = val_metrics["AUC"]
            bad_epochs = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_metrics": val_metrics},
                       ckpt_dir / "trusttour_best.pth")
            print(f"  -> saved new best checkpoint (AUC={best_auc:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (no AUC improvement for {patience} epochs).")
                break

    torch.save({"model_state": model.state_dict(), "epoch": epoch}, ckpt_dir / "trusttour_last.pth")
    print("Training complete.")


if __name__ == "__main__":
    main()
