# utils/builders.py
import os
import argparse
from typing import Tuple, Any, Dict

import inspect
import torch
import torch.utils.data
from clip import clip

from dataloader.video_dataloader import train_data_loader, test_data_loader
from models.Generate_Model import GenerateModel

# ✅ alias để tránh recursion
from models.Text import get_class_info as text_get_class_info


def get_class_info(args: argparse.Namespace) -> Tuple[list, list]:
    return text_get_class_info(args)


def build_model(args: argparse.Namespace, input_text: list) -> torch.nn.Module:
    device = args.device  # torch.device("mps") / "cuda" / "cpu"
    CLIP_model, _ = clip.load(args.clip_path, device=device, jit=False)
    # Explicitly convert CLIP model to float32 to avoid MPS datatype issues
    CLIP_model.float()

    model = GenerateModel(input_text=input_text, clip_model=CLIP_model, args=args)

    # freeze all
    for _, p in model.named_parameters():
        p.requires_grad = False

    trainable_keywords = ["image_encoder", "temporal_net", "prompt_learner", "temporal_net_body", "project_fc"]
    print("\nTrainable parameters:")
    for name, p in model.named_parameters():
        if any(k in name for k in trainable_keywords):
            p.requires_grad = True
            print(f"- {name}")
    print("************************\n")
    return model


def _split_dataset_and_collate(retval: Any):
    """
    normalize return of loader:
      - dataset
      - (dataset, collate_fn)
    """
    if isinstance(retval, tuple) and len(retval) == 2:
        return retval[0], retval[1]
    return retval, None


def _filter_kwargs_for_fn(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only kwargs that appear in fn signature.
    This prevents: unexpected keyword argument 'data_percentage' etc.
    """
    sig = inspect.signature(fn)
    accepted = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in accepted}


def _call_loader(fn, **kwargs):
    safe_kwargs = _filter_kwargs_for_fn(fn, kwargs)
    return fn(**safe_kwargs)


def build_dataloaders(args: argparse.Namespace):
    project_root = os.getcwd()

    # root_dir: allow relative or absolute
    root_dir = getattr(args, "root_dir", ".")
    root_dir_full = root_dir if os.path.isabs(root_dir) else os.path.join(project_root, root_dir)

    # annotation files: allow relative or absolute
    train_ann = args.train_annotation
    val_ann   = getattr(args, "val_annotation", "")
    test_ann  = args.test_annotation

    if not os.path.isabs(train_ann):
        train_ann = os.path.join(project_root, train_ann)
    if val_ann and (not os.path.isabs(val_ann)):
        val_ann = os.path.join(project_root, val_ann)
    if not os.path.isabs(test_ann):
        test_ann = os.path.join(project_root, test_ann)

    # bbox files: allow relative or absolute
    bbox_face = getattr(args, "bounding_box_face", "")
    bbox_body = getattr(args, "bounding_box_body", "")

    if bbox_face and (not os.path.isabs(bbox_face)):
        bbox_face = os.path.join(project_root, bbox_face)
    if bbox_body and (not os.path.isabs(bbox_body)):
        bbox_body = os.path.join(project_root, bbox_body)

    # optional
    data_percentage = float(getattr(args, "data_percentage", 1.0))

    # ---------- TRAIN ----------
    train_ret = _call_loader(
        train_data_loader,
        list_file=train_ann,
        num_segments=args.num_segments,
        duration=args.duration,
        image_size=args.image_size,
        dataset_name=args.dataset,
        bounding_box_face=bbox_face,
        bounding_box_body=bbox_body,
        root_dir=root_dir_full,
        data_percentage=data_percentage,   # ✅ will be auto-dropped if not supported
    )
    train_data, train_collate_fn = _split_dataset_and_collate(train_ret)

    # ---------- VAL ----------
    val_list_file = val_ann if val_ann else test_ann
    val_ret = _call_loader(
        test_data_loader,
        list_file=val_list_file,
        num_segments=args.num_segments,
        duration=args.duration,
        image_size=args.image_size,
        bounding_box_face=bbox_face,
        bounding_box_body=bbox_body,
        root_dir=root_dir_full,
        data_percentage=data_percentage,   # ✅ will be auto-dropped if not supported
    )
    val_data, val_collate_fn = _split_dataset_and_collate(val_ret)

    # ---------- TEST ----------
    test_ret = _call_loader(
        test_data_loader,
        list_file=test_ann,
        num_segments=args.num_segments,
        duration=args.duration,
        image_size=args.image_size,
        bounding_box_face=bbox_face,
        bounding_box_body=bbox_body,
        root_dir=root_dir_full,
        data_percentage=data_percentage,   # ✅ will be auto-dropped if not supported
    )
    test_data, test_collate_fn = _split_dataset_and_collate(test_ret)


        # ---------- Split stats ----------
    class_names, _ = get_class_info(args)
    num_classes = len(class_names)

    if hasattr(train_data, "class_counts"):
        train_counts = train_data.class_counts(num_classes=num_classes, normalize=True)
        val_counts   = val_data.class_counts(num_classes=num_classes, normalize=True)
        test_counts  = test_data.class_counts(num_classes=num_classes, normalize=True)

        print("Train counts:", train_counts.tolist())
        print("Val counts:",   val_counts.tolist())
        print("Test counts:",  test_counts.tolist())

        if min(val_counts.tolist()) < 3:
            print(f"⚠️ WARNING: Val has very few samples in some class: {val_counts.tolist()}")
    else:
        print("⚠️ Dataset has no class_counts(). Please add it to VideoDataset.")

    # ---------- WeightedRandomSampler (optional) ----------
    use_wrs = str(getattr(args, "use_weighted_sampler", "False")) == "True"
    sampler = None
    if use_wrs:
        if hasattr(train_data, "get_labels"):
            labels_for_sampler = train_data.get_labels(normalize=True)
        else:
            labels_for_sampler = [int(r.label) - 1 for r in train_data.video_list]

        import numpy as np
        labels_np = np.asarray(labels_for_sampler, dtype=np.int64)
        counts = np.bincount(labels_np, minlength=num_classes).astype(np.float32)
        counts[counts == 0] = 1.0

        class_weights = counts.sum() / counts
        class_weights = class_weights / class_weights.mean()
        max_w = float(getattr(args, "max_class_weight", 10.0))
        class_weights = np.clip(class_weights, 0.0, max_w)

        sample_weights = torch.as_tensor(class_weights[labels_np], dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        print(f"=> Using WeightedRandomSampler | counts={counts.astype(int).tolist()} | class_w={np.round(class_weights, 3).tolist()}")

    # ---------- DataLoaders ----------
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=sampler,      # <-- dùng sampler
        shuffle=False,        # <-- MUST be False
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=train_collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=val_collate_fn
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=test_collate_fn
    )

    
    
    return train_loader, val_loader, test_loader
