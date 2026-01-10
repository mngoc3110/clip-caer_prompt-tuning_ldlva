# utils/utils.py
import os
import time
import logging
from typing import Optional

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


# =========================
# Meter utils
# =========================
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        val = float(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", log_txt_path: Optional[str] = None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.log_txt_path = log_txt_path

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        msg = "\t".join(entries)

        print(msg)
        if self.log_txt_path is not None:
      
            os.makedirs(os.path.dirname(self.log_txt_path), exist_ok=True)
            with open(self.log_txt_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


# =========================
# RecorderMeter (TRAIN only)
# =========================
class RecorderMeter(object):
    """
    Store TRAIN loss / WAR / UAR per epoch.
    (VAL removed – paper-style TRAIN/TEST only)
    """
    def __init__(self, total_epoch):
        self.total_epoch = total_epoch
        self.reset(total_epoch)

    def reset(self, total_epoch=None):
        if total_epoch is not None:
            self.total_epoch = total_epoch

        self.epoch = 0
        self.train_loss = []
        self.train_war = []
        self.train_uar = []

    def update(self, epoch, train_loss, train_war, train_uar):
        self.epoch = epoch
        self.train_loss.append(float(train_loss))
        self.train_war.append(float(train_war))
        self.train_uar.append(float(train_uar))


# =========================
# Checkpoint utils
# =========================
def save_checkpoint(state: dict, filename: str):
    """
    Save checkpoint to `filename`.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(path: str, map_location="cpu"):
    """
    Safe checkpoint loading (PyTorch >=2.6 compatible).
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)


# =========================
# Metrics (WAR / UAR) + CM + report
# =========================
def _unwrap_logits(model_out):
    """
    Model forward may return:
      - Tensor
      - (logits, ...)
      - dict with 'logits' / 'output'
    """
    if isinstance(model_out, torch.Tensor):
        return model_out

    if isinstance(model_out, (list, tuple)) and len(model_out) > 0:
        if isinstance(model_out[0], torch.Tensor):
            return model_out[0]

    if isinstance(model_out, dict):
        for k in ["logits", "output", "pred", "yhat"]:
            if k in model_out and isinstance(model_out[k], torch.Tensor):
                return model_out[k]

    raise TypeError(f"Cannot unwrap logits from type={type(model_out)}")


@torch.no_grad()
def computer_uar_war(
    model,
    device=None,
    test_loader=None,
    data_loader=None,
    class_names=None,
    desc="METRICS",
    log_txt_path: Optional[str] = None,
):
    """
    TRAIN / TEST ONLY.
    Priority: test_loader > data_loader
    """

    # ---- choose loader ----
    if test_loader is not None:
        loader = test_loader
        split_name = "TEST"
    elif data_loader is not None:
        loader = data_loader
        split_name = "DATA"
    else:
        raise TypeError("computer_uar_war: need test_loader or data_loader")

    # ---- device ----
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    all_preds, all_targets = [], []

    start = time.time()
    for images_face, images_body, target in loader:
        images_face = images_face.to(device, non_blocking=True)
        images_body = images_body.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        out = model(images_face, images_body)
        logits = _unwrap_logits(out)

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.detach().cpu())
        all_targets.append(target.detach().cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    cm = confusion_matrix(all_targets, all_preds)

    war = (cm.diagonal().sum() / max(cm.sum(), 1)) * 100.0
    per_class_recall = cm.diagonal() / (cm.sum(axis=1) + 1e-12)
    uar = float(np.mean(per_class_recall)) * 100.0

    if class_names is None:
        report = classification_report(all_targets, all_preds, digits=4)
    else:
        report = classification_report(
            all_targets, all_preds, target_names=class_names, digits=4
        )

    elapsed = time.time() - start
    msg = f"[{split_name}][{desc}] time={elapsed:.2f}s | WAR={war:.3f}% | UAR={uar:.3f}%"
    logging.info(msg)

    # ---- PRINT ----
    print("\n" + msg)
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification report:")
    print(report)

    # ---- LOG ----
    if log_txt_path:
        with open(log_txt_path, "a", encoding="utf-8") as f:
            f.write("\n" + msg + "\n")
            f.write("Confusion Matrix:\n")
            f.write(str(cm) + "\n")
            f.write("Classification report:\n")
            f.write(report + "\n")

    return war, uar, cm, report


# =========================
# LR helper
# =========================
def get_lr(optimizer):
    return [pg["lr"] for pg in optimizer.param_groups]
