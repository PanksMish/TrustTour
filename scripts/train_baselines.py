"""
Train a single baseline model (Xception, F3Net, Swin, AIDE, or IAPL) under
the same protocol as TrustTour for fair comparison (Section 5.2/5.3).

Usage:
    python scripts/train_baselines.py --config configs/baselines.yaml \
        --dataset cifake --root data/CIFAKE --model xception
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
from models.baselines.registry import build_model
from utils.logger import CSVLogger, print_metrics
from utils.losses import bce_loss
from utils.metrics import detection_metrics
from utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/baselines.yaml")
    p.add_argument("--model", type=str, required=True,
                    choices=["xception", "f3net", "swin", "aide", "iapl"])
    p.add_argument("--root", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None, choices=["cifake", "synthbuster"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_samples_per_class", type=int, default=None)
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_epoch(model, loader, device, optimizer=None, desc="train"):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, n = 0.0, 0
    all_labels, all_probs = [], []

    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        image = batch["image"].to(device)
        label = batch["label"].to(device)

        with torch.set_grad_enabled(is_train):
            prob, logit = model(image)
            loss = bce_loss(logit, label)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        bs = image.shape[0]
        total_loss += loss.item() * bs
        n += bs
        all_labels.append(label.detach().cpu().numpy())
        all_probs.append(prob.detach().cpu().numpy())
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
    root = args.root or cfg["data"]["root"]
    dataset = args.dataset or cfg["data"]["dataset"]
    img_size = cfg["img_size"]
    nw = cfg["data"].get("num_workers", 4)

    train_ds = build_dataset(root, dataset, "train", img_size, tourism=False,
                              max_samples_per_class=args.max_samples_per_class)
    test_ds = build_dataset(root, dataset, "test", img_size, tourism=False,
                             max_samples_per_class=args.max_samples_per_class)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                               num_workers=nw, drop_last=True)
    val_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=nw)

    model_kwargs = cfg.get("models", {}).get(args.model, {})
    model = build_model(args.model, **model_kwargs).to(device)

    opt_cfg = cfg["optimizer"]
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["scheduler"]["t_max"])

    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(cfg["output"]["log_dir"], filename=f"{args.model}_train_log.csv")

    best_auc = -1.0
    patience = cfg.get("early_stopping_patience", 10)
    bad_epochs = 0

    for epoch in range(1, cfg["epochs"] + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer, desc=f"{args.model} train e{epoch}")
        val_metrics = run_epoch(model, val_loader, device, optimizer=None, desc=f"{args.model} val e{epoch}")
        scheduler.step()

        print_metrics(f"{args.model} epoch {epoch} train", train_metrics)
        print_metrics(f"{args.model} epoch {epoch} val", val_metrics)
        logger.log({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        if val_metrics["AUC"] > best_auc:
            best_auc = val_metrics["AUC"]
            bad_epochs = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_metrics": val_metrics},
                       ckpt_dir / f"{args.model}_best.pth")
            print(f"  -> saved new best checkpoint (AUC={best_auc:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    print(f"Training complete for {args.model}.")


if __name__ == "__main__":
    main()
