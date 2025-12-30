# builders.py

import argparse
from typing import Tuple
import os
import torch
import torch.utils.data
from clip import clip

from dataloader.video_dataloader import train_data_loader, test_data_loader
from models.Generate_Model import GenerateModel
from models.Text import *
from utils.utils import *


def build_model(args: argparse.Namespace, input_text: list) -> torch.nn.Module:
    print("Loading pretrained CLIP model...")
    # clip.load expects model name (e.g., "ViT-B/16") or a path to a .pt file.
    # If args.clip_path contains a slash, it's treated as a model name.
    # Otherwise, it's treated as a local path to a .pt file.
    if '/' in args.clip_path: # e.g., "ViT-B/16"
        CLIP_model, _ = clip.load(args.clip_path, device='cpu')
    else: # e.g., "models/ViT-B-16.pt" or "path/to/ViT-B-32.pt"
        # Assuming args.clip_path is a local path to the actual .pt file
        # We need to construct the full path if root_dir is used for clip_path
        # However, CLIP expects direct model names for its default loading.
        # If it's a local file, it should be passed directly to clip.load
        # The user's original train.sh had ViT-B/32, which is a model name.
        # If it's a full path to a downloaded .pt file, clip.load might handle it.
        # For now, let's assume it's a model name like ViT-B/16.
        # If args.clip_path is a local file, it usually looks like "ViT-B-32.pt" in the models/clip folder.
        # The build_model function should handle the loading of the CLIP model from the path provided
        # or from its identifier. Let's make sure it handles the case where it's a local file in root_dir.
        
        # Checking if it's a model name or a path that needs to be joined with root_dir
        # A simple check: if it doesn't contain a slash, it might be a local filename.
        # CLIP's load function typically takes a model name (e.g., "ViT-B/16") or a local path to a .pt file.
        # The earlier change was `--clip-path ViT-B/16`. This is a model name, not a file path.
        # If it was a local file, the original path in train.sh was /media/D/zlm/code/single_four/models/ViT-B-32.pt
        # Let's revert to the assumption that args.clip_path is a model identifier unless proven otherwise.
        CLIP_model, _ = clip.load(args.clip_path, device='cpu')


    print("\nInput Text Prompts:")
    for text in input_text:
        print(text)

    print("\nInstantiating GenerateModel...")
    model = GenerateModel(input_text=input_text, clip_model=CLIP_model, args=args)

    for name, param in model.named_parameters():
        param.requires_grad = False

    trainable_params_keywords = ["image_encoder", "temporal_net", "prompt_learner", "temporal_net_body", "project_fc"]
    print('\nTrainable parameters:')
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in trainable_params_keywords):
            param.requires_grad = True
            print(f"- {name}")
    print('************************\n')

    return model


def get_class_info(args: argparse.Namespace) -> Tuple[list, list]:
    """
    根据数据集和文本类型获取 class_names 和 input_text（用于生成 CLIP 模型文本输入）。

    Returns:
        class_names: 类别名称，用于混淆矩阵等
        input_text: 输入文本，用于传入模型
    """
    if args.dataset == "RAER":
        class_names = ['Neutrality', 'Enjoyment', 'Confusion', 'Fatigue', 'Distraction.']
        class_names_with_context = class_names_with_context_5
        class_descriptor = class_descriptor_5
    else:
        raise NotImplementedError(f"Dataset '{args.dataset}' is not implemented yet.")

    if args.text_type == "class_names":
        input_text = class_names
    elif args.text_type == "class_names_with_context":
        input_text = class_names_with_context
    elif args.text_type == "class_descriptor":
        input_text = class_descriptor
    else:
        raise ValueError(f"Unknown text_type: {args.text_type}")

    return class_names, input_text



def build_dataloaders(args: argparse.Namespace) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: 
    train_annotation_file_path = os.path.join(args.root_dir, args.train_annotation)
    test_annotation_file_path = os.path.join(args.root_dir, args.test_annotation)
    
    # Correctly join root_dir with bounding box paths
    bounding_box_face_path = os.path.join(args.root_dir, args.bounding_box_face)
    bounding_box_body_path = os.path.join(args.root_dir, args.bounding_box_body)

    print("Loading train data...")
    train_data, train_collate_fn = train_data_loader(
        list_file=train_annotation_file_path, num_segments=args.num_segments,
        duration=args.duration, image_size=args.image_size,dataset_name=args.dataset,
        bounding_box_face=bounding_box_face_path,bounding_box_body=bounding_box_body_path,
        root_dir=args.root_dir, data_percentage=args.data_percentage
    )
    
    print("Loading test data...")
    test_data, test_collate_fn = test_data_loader(
        list_file=test_annotation_file_path, num_segments=args.num_segments,
        duration=args.duration, image_size=args.image_size,
        bounding_box_face=bounding_box_face_path,bounding_box_body=bounding_box_body_path,
        root_dir=args.root_dir, data_percentage=args.data_percentage
    )

    print("Creating DataLoader instances...")
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        collate_fn=train_collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        collate_fn=test_collate_fn
    )
    
    return train_loader, val_loader