"""
VL-JEPA Training Script
========================
Vision-Language Joint Embedding Predictive Architecture için training pipeline.

Kullanım:
    python src/train.py --config configs/default.yaml
    python src/train.py --epochs 100 --batch_size 32 --lr 1e-4
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.cuda.amp import autocast, GradScaler

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models.utils import VL_JEPA, GrassmannEncoder


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ModelConfig:
    """Model configuration."""
    vision_input_dim: int = 768
    vocab_size: int = 32000
    d_model: int = 256
    r: int = 32
    x_encoder_layers: int = 4
    y_encoder_layers: int = 4
    predictor_layers: int = 5
    window_offsets: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 12, 16])
    max_visual_tokens: int = 256
    max_text_tokens: int = 512
    dropout: float = 0.1
    alignment_temp: float = 0.07


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Training params
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    gradient_clip: float = 1.0
    
    # Scheduler
    scheduler: str = "cosine"  # "cosine" or "onecycle"
    min_lr: float = 1e-6
    
    # Mixed precision
    use_amp: bool = True
    
    # Checkpointing
    save_every: int = 5
    checkpoint_dir: str = "checkpoints"
    
    # Logging
    log_every: int = 10
    eval_every: int = 1
    
    # Data
    num_workers: int = 4
    pin_memory: bool = True
    
    # Reproducibility
    seed: int = 42


@dataclass
class Config:
    """Full configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(
            model=ModelConfig(**data.get('model', {})),
            training=TrainingConfig(**data.get('training', {}))
        )


# ============================================================================
# Dataset
# ============================================================================

class SyntheticVLDataset(Dataset):
    """
    Synthetic dataset for testing VL-JEPA training.
    Replace this with your actual dataset.
    """
    
    def __init__(
        self,
        num_samples: int = 10000,
        num_patches: int = 196,  # 14x14 patches from ViT
        vision_dim: int = 768,
        query_len: int = 32,
        target_len: int = 64,
        vocab_size: int = 32000
    ):
        self.num_samples = num_samples
        self.num_patches = num_patches
        self.vision_dim = vision_dim
        self.query_len = query_len
        self.target_len = target_len
        self.vocab_size = vocab_size
        
        # Pre-generate data for reproducibility
        torch.manual_seed(42)
        self.visual_features = torch.randn(num_samples, num_patches, vision_dim)
        self.query_ids = torch.randint(0, vocab_size, (num_samples, query_len))
        self.target_ids = torch.randint(0, vocab_size, (num_samples, target_len))
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            'visual_features': self.visual_features[idx],
            'query_ids': self.query_ids[idx],
            'target_ids': self.target_ids[idx]
        }


class RealVLDataset(Dataset):
    """
    Template for real VL dataset.
    Implement this for your actual data.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        vision_encoder: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
        max_query_len: int = 32,
        max_target_len: int = 64
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.vision_encoder = vision_encoder
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.max_target_len = max_target_len
        
        # Load your data here
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Dict]:
        """Load samples from data directory."""
        # Implement your data loading logic here
        # Example structure:
        # [
        #     {"image_path": "...", "query": "...", "target": "..."},
        #     ...
        # ]
        samples = []
        
        # TODO: Load your actual data
        # data_file = self.data_dir / f"{self.split}.json"
        # if data_file.exists():
        #     with open(data_file) as f:
        #         samples = json.load(f)
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # TODO: Implement your data processing
        # - Load and encode image with vision_encoder
        # - Tokenize query and target with tokenizer
        
        return {
            'visual_features': torch.zeros(196, 768),  # Placeholder
            'query_ids': torch.zeros(self.max_query_len, dtype=torch.long),
            'target_ids': torch.zeros(self.max_target_len, dtype=torch.long)
        }


# ============================================================================
# Training Utilities
# ============================================================================

class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger:
    """Simple logger for training metrics."""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.log_dir / "training.log"
        self.metrics_file = self.log_dir / "metrics.jsonl"
        
    def log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
    
    def log_metrics(self, epoch: int, metrics: Dict[str, float], split: str = "train"):
        data = {
            "epoch": epoch,
            "split": split,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(data) + '\n')


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# Training Loop
# ============================================================================

class VLJEPATrainer:
    """Trainer for VL-JEPA model."""
    
    def __init__(
        self,
        model: VL_JEPA,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: Config,
        device: torch.device,
        logger: Logger
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.logger = logger
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Scheduler
        total_steps = len(train_loader) * config.training.epochs
        warmup_steps = len(train_loader) * config.training.warmup_epochs
        
        if config.training.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=config.training.min_lr
            )
        else:
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=config.training.learning_rate,
                total_steps=total_steps,
                pct_start=config.training.warmup_epochs / config.training.epochs
            )
        
        # Mixed precision
        self.scaler = GradScaler() if config.training.use_amp else None
        
        # Tracking
        self.best_val_loss = float('inf')
        self.current_epoch = 0
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        loss_meter = AverageMeter()
        alignment_meter = AverageMeter()
        reg_meter = AverageMeter()
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            visual_features = batch['visual_features'].to(self.device)
            query_ids = batch['query_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.config.training.use_amp:
                with autocast():
                    output = self.model(visual_features, query_ids, target_ids)
                    loss = output['loss']
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(visual_features, query_ids, target_ids)
                loss = output['loss']
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip
                )
                
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Update meters
            batch_size = visual_features.size(0)
            loss_meter.update(loss.item(), batch_size)
            alignment_meter.update(output['alignment_loss'].item(), batch_size)
            reg_meter.update(output['reg_loss'].item(), batch_size)
            
            # Logging
            if (batch_idx + 1) % self.config.training.log_every == 0:
                lr = self.optimizer.param_groups[0]['lr']
                self.logger.log(
                    f"Epoch [{self.current_epoch+1}/{self.config.training.epochs}] "
                    f"Step [{batch_idx+1}/{len(self.train_loader)}] "
                    f"Loss: {loss_meter.avg:.4f} "
                    f"Align: {alignment_meter.avg:.4f} "
                    f"Reg: {reg_meter.avg:.4f} "
                    f"LR: {lr:.2e}"
                )
        
        elapsed = time.time() - start_time
        
        metrics = {
            'loss': loss_meter.avg,
            'alignment_loss': alignment_meter.avg,
            'reg_loss': reg_meter.avg,
            'elapsed_time': elapsed,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        loss_meter = AverageMeter()
        alignment_meter = AverageMeter()
        reg_meter = AverageMeter()
        
        for batch in self.val_loader:
            visual_features = batch['visual_features'].to(self.device)
            query_ids = batch['query_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            
            if self.config.training.use_amp:
                with autocast():
                    output = self.model(visual_features, query_ids, target_ids)
            else:
                output = self.model(visual_features, query_ids, target_ids)
            
            batch_size = visual_features.size(0)
            loss_meter.update(output['loss'].item(), batch_size)
            alignment_meter.update(output['alignment_loss'].item(), batch_size)
            reg_meter.update(output['reg_loss'].item(), batch_size)
        
        metrics = {
            'loss': loss_meter.avg,
            'alignment_loss': alignment_meter.avg,
            'reg_loss': reg_meter.avg
        }
        
        return metrics
    
    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config)
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = path.replace('.pt', '_best.pt')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.logger.log(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self):
        """Full training loop."""
        self.logger.log(f"Starting training on {self.device}")
        self.logger.log(f"Model parameters: {count_parameters(self.model):,}")
        
        checkpoint_dir = Path(self.config.training.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(self.current_epoch, self.config.training.epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            self.logger.log(
                f"Epoch {epoch+1} Train - "
                f"Loss: {train_metrics['loss']:.4f} "
                f"Align: {train_metrics['alignment_loss']:.4f} "
                f"Reg: {train_metrics['reg_loss']:.4f} "
                f"Time: {train_metrics['elapsed_time']:.1f}s"
            )
            self.logger.log_metrics(epoch, train_metrics, "train")
            
            # Validate
            if (epoch + 1) % self.config.training.eval_every == 0:
                val_metrics = self.validate()
                if val_metrics:
                    self.logger.log(
                        f"Epoch {epoch+1} Val - "
                        f"Loss: {val_metrics['loss']:.4f} "
                        f"Align: {val_metrics['alignment_loss']:.4f} "
                        f"Reg: {val_metrics['reg_loss']:.4f}"
                    )
                    self.logger.log_metrics(epoch, val_metrics, "val")
                    
                    # Check if best
                    is_best = val_metrics['loss'] < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_metrics['loss']
                        self.logger.log(f"New best validation loss: {self.best_val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.training.save_every == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
                self.save_checkpoint(str(checkpoint_path), is_best=is_best if val_metrics else False)
                self.logger.log(f"Saved checkpoint to {checkpoint_path}")
        
        # Save final model
        final_path = checkpoint_dir / "final_model.pt"
        self.save_checkpoint(str(final_path))
        self.logger.log(f"Training complete! Final model saved to {final_path}")


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train VL-JEPA model")
    
    # Config
    parser.add_argument('--config', type=str, help='Path to config file')
    
    # Model
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--r', type=int, default=32)
    parser.add_argument('--x_encoder_layers', type=int, default=4)
    parser.add_argument('--y_encoder_layers', type=int, default=4)
    parser.add_argument('--predictor_layers', type=int, default=5)
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    
    # Data
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--num_samples', type=int, default=10000, help='Synthetic dataset size')
    
    # Misc
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load or create config
    if args.config and Path(args.config).exists():
        config = Config.load(args.config)
    else:
        config = Config()
    
    # Override with command line args
    config.model.d_model = args.d_model
    config.model.r = args.r
    config.model.x_encoder_layers = args.x_encoder_layers
    config.model.y_encoder_layers = args.y_encoder_layers
    config.model.predictor_layers = args.predictor_layers
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.training.weight_decay = args.weight_decay
    config.training.checkpoint_dir = args.checkpoint_dir
    config.training.seed = args.seed
    
    # Setup
    set_seed(config.training.seed)
    device = get_device()
    
    # Logger
    log_dir = Path(config.training.checkpoint_dir) / "logs"
    logger = Logger(str(log_dir))
    
    logger.log("=" * 60)
    logger.log("VL-JEPA Training")
    logger.log("=" * 60)
    logger.log(f"Device: {device}")
    logger.log(f"Config: {json.dumps(asdict(config), indent=2)}")
    
    # Dataset
    if args.data_dir:
        # Use real dataset
        train_dataset = RealVLDataset(args.data_dir, split="train")
        val_dataset = RealVLDataset(args.data_dir, split="val")
    else:
        # Use synthetic dataset
        logger.log(f"Using synthetic dataset with {args.num_samples} samples")
        full_dataset = SyntheticVLDataset(
            num_samples=args.num_samples,
            vision_dim=config.model.vision_input_dim,
            vocab_size=config.model.vocab_size
        )
        
        # Split into train/val
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )
    
    logger.log(f"Train samples: {len(train_dataset)}")
    logger.log(f"Val samples: {len(val_dataset)}")
    
    # Model
    model = VL_JEPA(
        vision_input_dim=config.model.vision_input_dim,
        vocab_size=config.model.vocab_size,
        d_model=config.model.d_model,
        r=config.model.r,
        x_encoder_layers=config.model.x_encoder_layers,
        y_encoder_layers=config.model.y_encoder_layers,
        predictor_layers=config.model.predictor_layers,
        window_offsets=config.model.window_offsets,
        max_visual_tokens=config.model.max_visual_tokens,
        max_text_tokens=config.model.max_text_tokens,
        dropout=config.model.dropout,
        alignment_temp=config.model.alignment_temp
    )
    
    logger.log(f"Model parameters: {count_parameters(model):,}")
    
    # Trainer
    trainer = VLJEPATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        logger=logger
    )
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Save config
    config_path = Path(config.training.checkpoint_dir) / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config.save(str(config_path))
    logger.log(f"Config saved to {config_path}")
    
    # Train!
    trainer.train()


if __name__ == "__main__":
    main()
