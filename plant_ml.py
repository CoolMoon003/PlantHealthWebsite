"""MobileNetV3-Small training/prediction helpers shared by train_model.py and
evaluate_group_cv.py.  Requires torch + torchvision.

Differences from the original scripts (each one is a deliberate fix):
  * Frozen layers are kept in eval() mode while training (``bn_eval_frozen``).
    The original code froze the backbone's *weights* but left its BatchNorm
    layers in train mode, so a "frozen" backbone still changed its running
    statistics and normalised with noisy 16-image batch statistics.
  * Images are decoded and resized to 224x224 once and cached, so each epoch
    costs only augmentation + forward/backward (the PNGs are 1024x1024).
    Resizing first and augmenting second matches the original transform order.
  * Early stopping monitors validation macro-F1 (or accuracy for the legacy
    replica), breaking ties with validation loss, and restores the best epoch.
  * Loaders are seeded; class weights are computed from the training split only.
"""
from __future__ import annotations

import copy
import random
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from groupcv import CLASSES, class_weights, classification_metrics

IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

DEFAULT_CONFIG = {
    "unfreeze_from": None,      # None = frozen backbone; k = train features[k:] (MobileNetV3-Small has 13 blocks)
    "bn_eval_frozen": True,     # keep frozen blocks in eval() mode while training
    "loss": "ce",               # "ce" or "weighted" (inverse-frequency, from the training split)
    "aug": "baseline",          # "none", "baseline" (flip+rot10), "mild_jitter" (+brightness/contrast 0.1), "robust"
    "sampler": "none",          # "none" or "weighted" (WeightedRandomSampler; mutually exclusive with loss='weighted')
    "label_smoothing": 0.0,     # CrossEntropyLoss label_smoothing epsilon; 0.0 = standard CE (off)
    "lr": 1e-3,                 # classifier head learning rate
    "backbone_lr": 1e-4,        # only used when unfreeze_from is set
    "weight_decay": 0.0,        # AdamW decoupled weight decay
    "batch_size": 16,
    "max_epochs": 15,
    "patience": 4,
    "monitor": "macro_f1",      # "macro_f1" or "accuracy"
    "pretrained": True,         # tests set False to avoid downloading ImageNet weights
}


def resolve_config(config: dict) -> dict:
    cfg = {**DEFAULT_CONFIG, **config}
    if cfg["monitor"] not in ("macro_f1", "accuracy"):
        raise ValueError("monitor must be 'macro_f1' or 'accuracy'")
    if cfg["loss"] not in ("ce", "weighted"):
        raise ValueError("loss must be 'ce' or 'weighted'")
    if cfg["aug"] not in ("none", "baseline", "mild_jitter", "robust"):
        raise ValueError("aug must be 'none', 'baseline', 'mild_jitter', or 'robust'")
    if cfg["sampler"] not in ("none", "weighted"):
        raise ValueError("sampler must be 'none' or 'weighted'")
    if not (0.0 <= cfg["label_smoothing"] < 1.0):
        raise ValueError("label_smoothing must be in [0.0, 1.0)")
    if cfg["sampler"] == "weighted" and cfg["loss"] == "weighted":
        raise ValueError(
            "sampler='weighted' and loss='weighted' must not be used together: "
            "the sampler already equalises per-batch class distribution, so "
            "the inverse-frequency loss weight would over-correct. "
            "Use loss='ce' when sampler='weighted'."
        )
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_image(path: str) -> Image.Image:
    """Decode, honour EXIF orientation, convert to RGB and resize to 224x224 (same as the app)."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return im.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)


def load_image_cache(records: Sequence[dict]) -> dict[str, Image.Image]:
    return {r["sha256"]: load_image(r["path"]) for r in records}


def make_transforms(aug: str) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    evaluate = transforms.Compose(normalize)
    if aug == "none":
        return evaluate, evaluate
    steps = [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)]
    if aug == "mild_jitter":
        steps.append(transforms.ColorJitter(brightness=0.1, contrast=0.1))
    elif aug == "robust":
        # Moderate augmentation designed for leaf symptom images.
        # Colour jitter: brightness/contrast/saturation 0.25 preserves diagnostic
        # yellowing and browning; hue ±0.04 is small enough not to confuse
        # chlorotic yellow with healthy green.
        # RandomPerspective simulates imperfect leaf angles without strongly
        # warping venation patterns (distortion 0.2, p=0.4).
        # RandomVerticalFlip: leaves have no fixed orientation.
        # RandomGrayscale (p=0.05): forces a small amount of colour-independent
        # feature learning. Kept very low so colour symptoms remain dominant.
        # No RandomErasing: erasing tissue would remove diagnostic symptom areas.
        steps = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(25),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.04),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.4),
            transforms.RandomGrayscale(p=0.05),
        ]
    return transforms.Compose(steps + normalize), evaluate


class RecordDataset(Dataset):
    def __init__(self, records: Sequence[dict], images: dict[str, Image.Image], transform):
        self.records = list(records)
        self.images = images
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        rec = self.records[index]
        return self.transform(self.images[rec["sha256"]]), CLASSES.index(rec["label"])


def build_model(config: dict) -> nn.Module:
    cfg = resolve_config(config)
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if cfg["pretrained"] else None
    model = models.mobilenet_v3_small(weights=weights)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    if cfg["unfreeze_from"] is not None:
        for block in list(model.features)[cfg["unfreeze_from"]:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(CLASSES))
    return model


def set_train_mode(model: nn.Module, bn_eval_frozen: bool) -> None:
    """model.train(), but blocks with no trainable parameter stay in eval() mode."""
    model.train()
    if bn_eval_frozen:
        for block in model.features:
            if not any(p.requires_grad for p in block.parameters()):
                block.eval()


def make_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    head = [p for p in model.classifier.parameters() if p.requires_grad]
    backbone = [p for p in model.features.parameters() if p.requires_grad]
    groups = [{"params": head, "lr": cfg["lr"]}]
    if backbone:
        groups.append({"params": backbone, "lr": cfg["backbone_lr"]})
    return torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])


@torch.no_grad()
def predict_probabilities(model: nn.Module, records: Sequence[dict], images: dict, device, batch_size: int = 32) -> np.ndarray:
    _, evaluate_tf = make_transforms("none")
    loader = DataLoader(RecordDataset(records, images, evaluate_tf), batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    chunks = [torch.softmax(model(x.to(device)), dim=1).cpu().numpy() for x, _ in loader]
    return np.concatenate(chunks) if chunks else np.zeros((0, len(CLASSES)))


def fit(
    train: Sequence[dict],
    val: Sequence[dict],
    config: dict,
    seed: int,
    images: dict[str, Image.Image],
    device: torch.device,
    log=None,
) -> tuple[nn.Module, dict]:
    """Train on ``train``; if ``val`` is non-empty use it for early stopping and restore the best epoch.

    With an empty ``val`` the model trains for exactly ``max_epochs`` epochs and the last epoch is kept.
    """
    cfg = resolve_config(config)
    set_seed(seed)
    train_tf, eval_tf = make_transforms(cfg["aug"])
    generator = torch.Generator()
    generator.manual_seed(seed)
    drop_last = len(train) % cfg["batch_size"] == 1  # avoid a final batch of one image

    train_dataset = RecordDataset(train, images, train_tf)
    if cfg["sampler"] == "weighted":
        # WeightedRandomSampler: give every class equal expected representation
        # per mini-batch by upsampling minority classes.
        # replacement=True is required (minority classes have fewer images than
        # a full epoch's worth of draws); num_samples=len(train) keeps epoch
        # length the same as a plain shuffle, so learning-rate schedules based
        # on epoch count remain comparable across configs.
        from torch.utils.data import WeightedRandomSampler
        label_indices = [CLASSES.index(r["label"]) for r in train]
        counts = [max(label_indices.count(i), 1) for i in range(len(CLASSES))]
        sample_weights = [1.0 / counts[li] for li in label_indices]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train),
            replacement=True,
            generator=generator,
        )
        train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], sampler=sampler,
                                  num_workers=0, drop_last=drop_last)
    else:
        train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True,
                                  generator=generator, num_workers=0, drop_last=drop_last)

    val_loader = DataLoader(RecordDataset(val, images, eval_tf), batch_size=32, shuffle=False, num_workers=0) if val else None

    model = build_model(cfg).to(device)
    optimizer = make_optimizer(model, cfg)
    weight = None
    if cfg["loss"] == "weighted":
        weight = torch.tensor(class_weights([r["label"] for r in train]), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg["label_smoothing"])
    plain_ce = nn.CrossEntropyLoss()

    best = {"score": -1.0, "loss": float("inf"), "state": None, "epoch": 0}
    stale = 0
    history = []
    epochs_run = 0
    for epoch in range(1, cfg["max_epochs"] + 1):
        set_train_mode(model, cfg["bn_eval_frozen"])
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
        epochs_run = epoch
        if val_loader is None:
            continue
        model.eval()
        outputs, targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                outputs.append(model(x.to(device)).cpu())
                targets.extend(y.tolist())
        logits = torch.cat(outputs)
        preds = logits.argmax(1).tolist()
        metrics = classification_metrics(targets, preds, present_only=True)
        val_loss = float(plain_ce(logits, torch.tensor(targets)))
        score = metrics[cfg["monitor"]]
        history.append({"epoch": epoch, "val_score": score, "val_loss": val_loss})
        if log:
            log(f"    epoch {epoch:2d} val_{cfg['monitor']}={score:.3f} val_loss={val_loss:.3f}")
        better = score > best["score"] + 1e-9 or (abs(score - best["score"]) <= 1e-9 and val_loss < best["loss"] - 1e-6)
        if better:
            best = {"score": score, "loss": val_loss, "state": copy.deepcopy(model.state_dict()), "epoch": epoch}
            stale = 0
        else:
            stale += 1
            if stale >= cfg["patience"]:
                break
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()
    return model, {"best_epoch": best["epoch"] if val_loader is not None else epochs_run,
                   "epochs_run": epochs_run, "history": history, "config": cfg}


def make_cv_trainer(images: dict[str, Image.Image], device: torch.device, log=None):
    """Adapter with the cv_runner Trainer signature."""
    def trainer(train, val, test, config, seed):
        model, info = fit(train, val, config, seed, images, device, log=log)
        test_prob = predict_probabilities(model, test, images, device)
        out = {
            "test_pred": test_prob.argmax(1).tolist(),
            "test_confidence": test_prob.max(1).tolist(),
            "best_epoch": info["best_epoch"],
            "epochs_run": info["epochs_run"],
        }
        if val:
            out["val_pred"] = predict_probabilities(model, val, images, device).argmax(1).tolist()
        return out
    return trainer
