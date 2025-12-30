# ==================== Imports ====================
import argparse
import datetime
import os
import random
import shutil
import time

import matplotlib
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import warnings
from clip import clip
from dataloader.video_dataloader import train_data_loader, test_data_loader
from models.Generate_Model import GenerateModel
from models.Text import *
from trainer import Trainer
from utils.loss import *
from utils.utils import *
from utils.builders import *

# Ignore specific warnings (for cleaner output)
warnings.filterwarnings("ignore", category=UserWarning)
# Use 'Agg' backend for matplotlib (no GUI required)
matplotlib.use('Agg')

# ==================== Argument Parser ====================
parser = argparse.ArgumentParser(
    description='A highly configurable training script for RAER Dataset',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# --- Experiment and Environment ---
exp_group = parser.add_argument_group('Experiment & Environment', 'Basic settings for the experiment')
exp_group.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                       help="Execution mode: 'train' for a full training run, 'eval' for evaluation only.")
exp_group.add_argument('--eval-checkpoint', type=str, default='/media/D/zlm/code/CLIP_CAER/outputs_1/test-[07-09]-[22:24]/model_best.pth',
                       help="Path to the model checkpoint for evaluation mode (e.g., outputs/exp_name/model_best.pth).")
exp_group.add_argument('--exper-name', type=str, default='test', help='A name for the experiment to create a unique output folder.')
exp_group.add_argument('--dataset', type=str, default='RAER', help='Name of the dataset to use.')
exp_group.add_argument('--gpu', type=str, default='0', help='ID of the GPU to use, or "mps"/"cpu".')
exp_group.add_argument('--workers', type=int, default=4, help='Number of data loading workers.')
exp_group.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')

# --- Data & Path ---
path_group = parser.add_argument_group('Data & Path', 'Paths to datasets and pretrained models')
path_group.add_argument('--root-dir', type=str, default='/media/F/FERDataset/AER-DB', help='Root directory of the dataset.')
path_group.add_argument('--train-annotation', type=str, default='RAER/annotation/train.txt', help='Path to training annotation file, relative to root-dir.')
path_group.add_argument('--test-annotation', type=str, default='RAER/test.txt', help='Path to testing annotation file, relative to root-dir.')
path_group.add_argument('--clip-path', type=str, default='/media/D/zlm/code/single_four/models/ViT-B-32.pt', help='Path to the pretrained CLIP model.')
path_group.add_argument('--bounding-box-face', type=str, default='/media/F/FERDataset/AER-DB/RAER/bounding_box/face_abs.json')
path_group.add_argument('--bounding-box-body', type=str, default="/media/F/FERDataset/AER-DB/RAER/bounding_box/body_abs.json")
path_group.add_argument('--data-percentage', type=float, default=1.0, help='Percentage of the dataset to use for training and validation (e.g., 0.1 for 10%).')

# --- Training Control ---
train_group = parser.add_argument_group('Training Control', 'Parameters to control the training process')
train_group.add_argument('--epochs', type=int, default=20, help='Total number of training epochs.')
train_group.add_argument('--batch-size', type=int, default=8, help='Batch size for training and validation.')
train_group.add_argument('--print-freq', type=int, default=10, help='Frequency of printing training logs.')

# --- Optimizer & Learning Rate ---
optim_group = parser.add_argument_group('Optimizer & LR', 'Hyperparameters for the optimizer and scheduler')
optim_group.add_argument('--lr', type=float, default=1e-2, help='Initial learning rate for main modules.')
optim_group.add_argument('--lr-image-encoder', type=float, default=1e-5, help='Learning rate for the image encoder part.')
optim_group.add_argument('--lr-prompt-learner', type=float, default=1e-3, help='Learning rate for the prompt learner.')
optim_group.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for the optimizer.')
optim_group.add_argument('--momentum', type=float, default=0.9, help='Momentum for the SGD optimizer.')
optim_group.add_argument('--milestones', nargs='+', type=int, default=[10, 15], help='Epochs at which to decay the learning rate.')
optim_group.add_argument('--gamma', type=float, default=0.1, help='Factor for learning rate decay.')

# --- Model & Input ---
model_group = parser.add_argument_group('Model & Input', 'Parameters for model architecture and data handling')
model_group.add_argument('--text-type', default='class_descriptor', choices=['class_names', 'class_names_with_context', 'class_descriptor'], help='Type of text prompts to use.')
model_group.add_argument('--temporal-layers', type=int, default=1, help='Number of layers in the temporal modeling part.')
model_group.add_argument('--contexts-number', type=int, default=8, help='Number of context vectors in the prompt learner.')
model_group.add_argument('--class-token-position', type=str, default="end", help='Position of the class token in the prompt.')
model_group.add_argument('--class-specific-contexts', type=str, default='True', choices=['True', 'False'], help='Whether to use class-specific context prompts.')
model_group.add_argument('--load_and_tune_prompt_learner', type=str, default='True', choices=['True', 'False'], help='Whether to load and fine-tune the prompt learner.')
model_group.add_argument('--num-segments', type=int, default=16, help='Number of segments to sample from each video.')
model_group.add_argument('--duration', type=int, default=1, help='Duration of each segment.')
model_group.add_argument('--image-size', type=int, default=224, help='Size to resize input images to.')

# --- Loss & Regularization ---
loss_group = parser.add_argument_group('Loss & Regularization', 'Hyperparameters for loss functions and regularization')
loss_group.add_argument('--lambda-mi', type=float, default=1.0, help='Weight for Mutual Information (MI) loss.')
loss_group.add_argument('--lambda-dc', type=float, default=1.0, help='Weight for Distance Correlation (DC) loss.')
loss_group.add_argument('--mi-warmup', type=int, default=0, help='Epochs to warmup MI loss.')
loss_group.add_argument('--mi-ramp', type=int, default=0, help='Epochs to ramp up MI loss.')
loss_group.add_argument('--dc-warmup', type=int, default=0, help='Epochs to warmup DC loss.')
loss_group.add_argument('--dc-ramp', type=int, default=0, help='Epochs to ramp up DC loss.')
loss_group.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing factor.')
loss_group.add_argument('--semantic-smoothing', type=str, default='True', choices=['True', 'False'], help='Whether to use semantic-guided label smoothing (LDLVA-inspired).')
loss_group.add_argument('--smoothing-temp', type=float, default=0.1, help='Temperature for semantic label distribution (lower = sharper).')
loss_group.add_argument('--use-amp', type=str, default='True', choices=['True', 'False'], help='Enable or disable Automatic Mixed Precision (AMP).')
loss_group.add_argument('--gradient-accumulation-steps', type=int, default=1, help='Number of steps to accumulate gradients before updating weights.')



# ==================== Helper Functions ====================
def setup_environment(args: argparse.Namespace) -> argparse.Namespace:
    if args.gpu == 'mps':
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            print("MPS not available, falling back to CPU.")
            device = torch.device("cpu")
    elif args.gpu == 'cpu':
        device = torch.device("cpu")
    elif torch.cuda.is_available() and args.gpu.isdigit():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        print(f"CUDA not available or invalid GPU ID '{args.gpu}', falling back to CPU.")
        device = torch.device("cpu")
    
    args.device = device

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True
    
    print("Environment and random seeds set successfully.")
    return args


def setup_paths_and_logging(args: argparse.Namespace) -> argparse.Namespace:
    now = datetime.datetime.now()
    time_str = now.strftime("-[%m-%d]-[%H:%M]")
    
    args.name = args.exper_name + time_str
        
    args.output_path = os.path.join("outputs", args.name)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    print('************************')
    print("Running with the following configuration:")
    for k, v in vars(args).items():
        print(f'{k} = {v}')
    print('************************')
    
    log_txt_path = os.path.join(args.output_path, 'log.txt')
    with open(log_txt_path, 'w') as f:
        for k, v in vars(args).items():
            f.write(f'{k} = {v}\n')
        f.write('*'*50 + '\n\n')
        
    return args

# ==================== Training Function ====================
def run_training(args: argparse.Namespace) -> None:
    # Paths for logging and saving
    log_txt_path = os.path.join(args.output_path, 'log.txt')
    log_curve_path = os.path.join(args.output_path, 'log.png')
    log_confusion_matrix_path = os.path.join(args.output_path, 'confusion_matrix.png')
    checkpoint_path = os.path.join(args.output_path, 'model.pth')
    best_checkpoint_path = os.path.join(args.output_path, 'model_best.pth')        
    best_uar = 0.0
    start_epoch = 0
    recorder = RecorderMeter(args.epochs)
    
    # Build model
    print("=> Building model...")
    class_names, input_text = get_class_info(args)
    model = build_model(args, input_text)
    model = model.to(args.device)
    print("=> Model built and moved to device successfully.")

    # Load data
    print("=> Building dataloaders...")
    train_loader, val_loader = build_dataloaders(args)
    print("=> Dataloaders built successfully.")

    # Calculate class weights for RAER to improve recall on minority classes
    if args.dataset == 'RAER':
        print("=> Calculating class weights for RAER to handle imbalance...")
        try:
            # Access the underlying dataset labels
            train_dataset = train_loader.dataset
            # Labels are 1-based in file, 0-based in training
            labels = [record.label - 1 for record in train_dataset.video_list]
            
            # Count frequencies
            class_counts = np.bincount(labels)
            total_samples = len(labels)
            num_classes = len(class_counts)
            
            # Compute weights: w_j = N / (K * n_j)
            # Add epsilon to avoid division by zero
            class_weights = total_samples / (num_classes * (class_counts + 1e-6))
            
            # Clip weights to prevent instability
            max_weight = 10.0
            print(f"   Clipping class weights to a maximum of {max_weight}")
            class_weights = np.clip(class_weights, a_min=None, a_max=max_weight)

            # Pass to args so build_criterion picks it up
            args.class_weights = class_weights.tolist()
            
            print(f"   Class Counts: {class_counts}")
            print(f"   Computed (and Clipped) Class Weights: {np.round(class_weights, 4)}")
            print("   (These weights will be used in the Loss function to penalize errors on minority classes more heavily)")
            
        except Exception as e:
            print(f"Warning: Could not calculate class weights automatically. Error: {e}")

    # Loss and optimizer
    criterion = build_criterion(args, mi_estimator=model.mi_estimator, num_classes=len(class_names)).to(args.device)
    optimizer = torch.optim.SGD([
        {"params": model.temporal_net.parameters(), "lr": args.lr},
        {"params": model.temporal_net_body.parameters(), "lr": args.lr},
        {"params": model.image_encoder.parameters(), "lr": args.lr_image_encoder},
        {"params": model.prompt_learner.parameters(), "lr": args.lr_prompt_learner},
        {"params": model.project_fc.parameters(), "lr": args.lr_image_encoder}
    ], momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    
    # Trainer
    trainer = Trainer(
        model, criterion, optimizer, scheduler, args.device, log_txt_path,
        use_amp=(args.use_amp == 'True'), 
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    best_war = 0.0 # Track best WAR

    for epoch in range(start_epoch, args.epochs):
        inf = f'******************** Epoch: {epoch} ********************'
        start_time = time.time()
        print(inf)
        with open(log_txt_path, 'a') as f:
            f.write(inf + '\n')

        # Log current learning rates
        current_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
        lr_str = ' '.join([f'{lr:.1e}' for lr in current_lrs])
        log_msg = f'Current learning rates: {lr_str}'
        with open(log_txt_path, 'a') as f:
            f.write(log_msg + '\n')
        print(log_msg)

        # Train & Validate
        train_war, train_uar, train_los, _ = trainer.train_epoch(train_loader, epoch)
        val_war, val_uar, val_los, _ = trainer.validate(val_loader, str(epoch))
        scheduler.step()

        # Update best metrics
        if val_uar > best_uar:
            best_uar = val_uar
        if val_war > best_war:
            best_war = val_war
        
        is_best = val_uar >= best_uar # Keep using UAR for is_best criterion or change as needed

        # Save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_uar,
            'best_war': best_war, 
            'optimizer': optimizer.state_dict(),
            'recorder': recorder
        }, checkpoint_path)
        
        if is_best:
            shutil.copyfile(checkpoint_path, best_checkpoint_path)

        # Record metrics
        epoch_time = time.time() - start_time
        # RecorderMeter in utils.py takes (epoch, train_loss, train_war, train_uar)
        recorder.update(epoch, train_los, train_war, train_uar)
        # recorder.plot_curve(log_curve_path) # plot_curve method is missing in utils.py

        # Log results
        log_msg = (
            f'Train WAR: {train_war:.2f}% | Train UAR: {train_uar:.2f}%\n'
            f'Valid WAR: {val_war:.2f}% | Valid UAR: {val_uar:.2f}%\n'
            f'Best Valid WAR: {best_war:.2f}% | Best Valid UAR: {best_uar:.2f}%\n'
            f'Epoch time: {epoch_time:.2f}s\n'
        )
        print(log_msg)
        with open(log_txt_path, 'a') as f:
            f.write(log_msg + '\n')

    # Final evaluation with best model
    pre_trained_dict = torch.load(best_checkpoint_path,map_location=args.device)['state_dict']
    model.load_state_dict(pre_trained_dict)
    computer_uar_war(
        val_loader=val_loader,
        model=model,
        device=args.device,
        class_names=class_names,
        log_confusion_matrix_path=log_confusion_matrix_path,
        log_txt_path=log_txt_path,
        title=f"Confusion Matrix on {args.dataset}"
    )

def run_eval(args: argparse.Namespace) -> None:
    print("=> Starting evaluation mode...")
    log_txt_path = os.path.join(args.output_path, 'log.txt')
    log_confusion_matrix_path = os.path.join(args.output_path, 'confusion_matrix.png')

    class_names, input_text = get_class_info(args)
    model = build_model(args, input_text)
    model = model.to(args.device)

    # Load pretrained weightsaaaaa
    ckpt = torch.load(args.eval_checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)

    # Load data
    _, val_loader = build_dataloaders(args)

    # Run evaluation
    computer_uar_war(
        test_loader=val_loader,
        model=model,
        device=args.device,
        class_names=class_names,
        log_txt_path=log_txt_path
    )
    print("=> Evaluation complete.")


# ==================== Entry Point ====================
if __name__ == '__main__':
    args = parser.parse_args()
    args = setup_environment(args)
    args = setup_paths_and_logging(args)
    
    if args.mode == 'eval':
        run_eval(args)
    else:
        run_training(args)