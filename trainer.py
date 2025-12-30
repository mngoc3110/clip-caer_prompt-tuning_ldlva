# trainer.py
import logging
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from utils.utils import AverageMeter, ProgressMeter

class Trainer:
    """A class that encapsulates the training and validation logic."""
    def __init__(self, model, criterion, optimizer, scheduler, device, log_txt_path, use_amp=True, gradient_accumulation_steps=1):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.print_freq = 10
        self.log_txt_path = log_txt_path
        
        # Store AMP and accumulation settings
        self.use_amp = use_amp
        self.accumulation_steps = gradient_accumulation_steps
        
        # Initialize GradScaler for AMP, only for CUDA devices
        # enabled flag handles both use_amp and device type check
        is_cuda = self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.use_amp and is_cuda))
        
        if self.use_amp and not is_cuda:
            print("AMP is enabled but device is not CUDA. GradScaler will be disabled. Autocast will proceed on MPS/CPU.")

    def _run_one_epoch(self, loader, epoch_str, is_train=True):
        """Runs one epoch of training or validation."""
        if is_train:
            self.model.train()
            # Zero gradients at the beginning of a training epoch
            self.optimizer.zero_grad()
            prefix = f"Train Epoch: [{epoch_str}]"
        else:
            self.model.eval()
            prefix = f"Valid Epoch: [{epoch_str}]"

        losses = AverageMeter('Loss', ':.4e')
        war_meter = AverageMeter('WAR', ':6.2f')
        progress = ProgressMeter(
            len(loader), 
            [losses, war_meter], 
            prefix=prefix, 
            log_txt_path=self.log_txt_path  
        )

        all_preds = []
        all_targets = []

        # Use torch.no_grad for validation, otherwise default context manager
        context = torch.enable_grad() if is_train else torch.no_grad()
        
        with context:
            for i, batch_data in enumerate(loader):
                # Handle potential empty batches
                if batch_data is None or (isinstance(batch_data, torch.Tensor) and batch_data.numel() == 0):
                    continue
                
                images_face, images_body, target = batch_data
                images_face = images_face.to(self.device)
                images_body = images_body.to(self.device)
                target = target.to(self.device)

                # Device-agnostic AMP autocasting
                with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                    output_dict = self.model(images_face, images_body)
                    
                    logits = output_dict["logits"]
                    
                    try:
                        epoch_num = int(epoch_str)
                    except ValueError:
                        epoch_num = 0

                    loss_dict = self.criterion(
                        logits, 
                        target, 
                        epoch=epoch_num,
                        learnable_text_features=output_dict.get("learnable_text_features"),
                        hand_crafted_text_features=output_dict.get("hand_crafted_text_features"),
                        logits_hand=output_dict.get("logits_hand")
                    )
                    loss = loss_dict["total"]
                    
                    # Normalize loss for gradient accumulation
                    if self.accumulation_steps > 1:
                        loss = loss / self.accumulation_steps

                if is_train:
                    # Backward pass
                    if self.device.type == 'cuda':
                        self.scaler.scale(loss).backward()
                    else: # for MPS or CPU
                        loss.backward()

                    # Optimizer step (only after designated accumulation steps)
                    if (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(loader):
                        if self.device.type == 'cuda':
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else: # for MPS or CPU
                            self.optimizer.step()
                        
                        self.optimizer.zero_grad()

                # Record metrics
                preds = logits.argmax(dim=1)
                correct_preds = preds.eq(target).sum().item()
                acc = (correct_preds / target.size(0)) * 100.0

                losses.update(loss.item() * self.accumulation_steps, target.size(0)) # Scale loss back up for logging
                war_meter.update(acc, target.size(0))

                all_preds.append(preds.cpu())
                all_targets.append(target.cpu())

                if i % self.print_freq == 0:
                    progress.display(i)
        
        # Calculate epoch-level metrics
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        cm = confusion_matrix(all_targets.numpy(), all_preds.numpy())
        war = war_meter.avg # Weighted Average Recall (WAR) is the overall accuracy
        
        # Unweighted Average Recall (UAR)
        class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-6)
        uar = np.nanmean(class_acc) * 100

        logging.info(f"{prefix} * WAR: {war:.3f} | UAR: {uar:.3f}")
        with open(self.log_txt_path, 'a') as f:
            f.write(f'{prefix} * WAR: {war:.3f} | UAR: {uar:.3f}\n')
        return war, uar, losses.avg, cm
        
    def train_epoch(self, train_loader, epoch_num):
        """Executes one full training epoch."""
        return self._run_one_epoch(train_loader, str(epoch_num), is_train=True)
    
    def validate(self, val_loader, epoch_num_str="Final"):
        """Executes one full validation run."""
        return self._run_one_epoch(val_loader, epoch_num_str, is_train=False)