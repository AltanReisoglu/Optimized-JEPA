import os
import sys
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from decord import VideoReader, cpu
from tqdm import tqdm
from typing import Optional, Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import from fixed encoder
from project_src.models.encoder import VLJEPA, VLJEPAConfig
from transformers import AutoTokenizer

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    print("Warning: 'datasets' library not available. Install with: pip install datasets")
    DATASETS_AVAILABLE = False
import subprocess
import os

def download_youtube_video(url: str, output_path: str) -> bool:
    """
    Download YouTube video as mp4.
    Returns True if successful.
    """
    try:
        cmd = [
            "yt-dlp",
            "-f", "mp4",
            "-o", output_path,
            url
        ]
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        return os.path.exists(output_path)
    except Exception:
        return False


class LocalVideoDataset(Dataset):
    """
    Dataset for local videos in assets folder
    """
    def __init__(
        self, 
        config: VLJEPAConfig, 
        video_root: str = "assets",
        max_samples: Optional[int] = None
    ):
        self.config = config
        self.video_root = video_root
        
        # Get all video files
        self.video_files = []
        if os.path.exists(video_root):
            for file in os.listdir(video_root):
                if file.endswith('.mp4'):
                    self.video_files.append(file)
        
        # Limit dataset size if specified
        if max_samples is not None:
            self.video_files = self.video_files[:max_samples]
        
        # Processor for V-JEPA
        print("Loading V-JEPA preprocessor...")
        try:
            self.processor = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_preprocessor')
            print("✓ Preprocessor loaded")
        except Exception as e:
            print(f"⚠️  Failed to load vjepa2_preprocessor: {e}")
            self.processor = None

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.predictor_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Track failed videos
        self.failed_videos = set()
        
        print(f"Dataset initialized with {len(self.video_files)} video files")

    def __len__(self):
        return len(self.video_files)

    def process_video(self, video_path: str, start_time: float = 0.0, end_time: float = 8.0) -> Optional[torch.Tensor]:
        """
        Process video file and return tensor
        
        Returns:
            Tensor of shape (3, T, H, W) or None if failed
        """
        if not os.path.exists(video_path):
            if video_path not in self.failed_videos:
                print(f"⚠️  Video not found: {video_path}")
                self.failed_videos.add(video_path)
            return None

        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            fps = vr.get_avg_fps()
            total_frames = len(vr)
            
            # Calculate frame indices
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            end_frame = min(end_frame, total_frames - 1)
            start_frame = max(0, start_frame)
            
            # Handle edge case
            if start_frame >= end_frame:
                start_frame = 0
                end_frame = total_frames - 1
            
            # Sample frames uniformly
            num_frames = self.config.VIDEO_INPUT_FRAMES
            if end_frame - start_frame < num_frames:
                indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
            else:
                indices = np.linspace(start_frame, end_frame, num_frames).astype(int)

            # Read frames
            frames = vr.get_batch(indices).asnumpy()
            
            # frames: (T, H, W, C) -> (T, C, H, W)
            # Normalize to [0, 1] IMMEDIATELY. This was missing!
            frames_tensor = torch.from_numpy(frames).float() / 255.0
            frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # (T, C, H, W)
            
            # Process with V-JEPA preprocessor if available
            # Note: Preprocessors often handle normalization, but feeding 0-255 usually breaks them if they expect 0-1.
            # We provide 0-1 float tensor now.
            if self.processor:
                try:
                    # Some processors expect (C, T, H, W) -> permute
                    frames_input = frames_tensor.permute(1, 0, 2, 3) # (C, T, H, W)
                    processed = self.processor(frames_input)
                    
                    # Handle different return types
                    if isinstance(processed, list): 
                        processed = processed[0]
                    elif isinstance(processed, dict):
                        if 'x' in processed:
                            processed = processed['x']
                        else:
                            processed = list(processed.values())[0]
                    
                    return processed
                except Exception as e:
                    if video_path not in self.failed_videos:
                        print(f"⚠️  Processor failed for {video_path} (falling back to manual): {e}")
                        # Don't add to failed_videos, fall through to manual
                    
            # Fallback: basic normalization
            # Ensure shape is (C, T, H, W)
            if frames_tensor.shape[1] == 3: # if (T, C, H, W)
                frames_tensor = frames_tensor.permute(1, 0, 2, 3)
                
            frames_tensor = F.interpolate(
                frames_tensor, 
                size=(self.config.VIDEO_INPUT_SIZE[0], self.config.VIDEO_INPUT_SIZE[1]),
                mode='bilinear',
                align_corners=False
            )
            # Already normalized to [0, 1] above
            
            return frames_tensor
            
        except Exception as e:
            if video_path not in self.failed_videos:
                print(f"⚠️  Error processing {video_path}: {e}")
                self.failed_videos.add(video_path)
            return None

    def __getitem__(self, idx: int) -> Dict:
        video_file = self.video_files[idx]
        video_path = os.path.join(self.video_root, video_file)
        video_id = video_file.replace('.mp4', '')
        
        # --- CAPTION LOADING LOGIC ---
        # Try to find a corresponding text file
        caption_path = os.path.join(self.video_root, video_file.replace('.mp4', '.txt'))
        caption = ""
        
        if os.path.exists(caption_path):
            try:
                with open(caption_path, 'r', encoding='utf-8') as f:
                    caption = f.read().strip()
            except Exception:
                pass
        
        # Fallback: Use filename as caption if no text file (better than empty)
        if not caption:
            # Clean up filename to make it a readable caption
            clean_name = video_id.replace('_', ' ').replace('-', ' ')
            caption = f"Video of {clean_name}"

        # --- PROCESS VIDEO ---
        pixel_values = self.process_video(video_path, start_time=0.0, end_time=8.0)

        if pixel_values is None:
            pixel_values = torch.zeros(
                3,
                self.config.VIDEO_INPUT_FRAMES,
                self.config.VIDEO_INPUT_SIZE[0],
                self.config.VIDEO_INPUT_SIZE[1]
            )

        # --- TOKENIZE QUERY ---
        prompt = "What is the topic of this video?"
        tokens = self.tokenizer(
            prompt,
            max_length=self.config.max_query_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "caption": caption,
            "video_id": video_id
        }


def bidirectional_infonce_loss(
    pred_embeddings: torch.Tensor, 
    target_embeddings: torch.Tensor, 
    temperature: float = 0.07
) -> torch.Tensor:
    """
    Bidirectional InfoNCE loss (symmetric contrastive loss)
    
    Args:
        pred_embeddings: (B, D) predicted embeddings
        target_embeddings: (B, D) target embeddings
        temperature: Temperature for scaling logits
    
    Returns:
        loss: Scalar loss value
    """
    # Normalize embeddings
    pred_embeddings = F.normalize(pred_embeddings, dim=-1)
    target_embeddings = F.normalize(target_embeddings, dim=-1)
    
    # Compute similarity matrix
    logits = torch.matmul(pred_embeddings, target_embeddings.t()) / temperature
    
    # Labels: diagonal elements are positive pairs
    labels = torch.arange(pred_embeddings.size(0), device=pred_embeddings.device)
    
    # Bidirectional loss
    loss1 = F.cross_entropy(logits, labels)
    loss2 = F.cross_entropy(logits.t(), labels)
    
    return (loss1 + loss2) / 2


def train_epoch(
    model: VLJEPA, 
    dataloader: DataLoader, 
    optimizer: torch.optim.Optimizer,
    device: str = 'cuda',
    use_amp: bool = True,
    scaler: Optional[GradScaler] = None,
    epoch: int = 0
) -> Dict[str, float]:
    """
    Train for one epoch
    
    Returns:
        Dictionary with training metrics
    """
    model.train()
    
    # Ensure x_encoder stays in eval mode (frozen)
    if hasattr(model, 'x_encoder'):
        model.x_encoder.eval()
    
    total_loss = 0
    num_batches = 0
    valid_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        try:
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            captions = batch['caption']  # List of strings
            
            # Skip batch if all videos failed to load
            if torch.all(pixel_values == 0):
                print(f"⚠️  Skipping batch {batch_idx} - all videos failed to load")
                continue
            
            optimizer.zero_grad()
            
            # Mixed precision training
            with autocast(enabled=use_amp):
                # 1. Encode video (frozen encoder)
                with torch.no_grad():
                    visual_features = model.x_encoder(pixel_values)
                    
                    # Handle dict output
                    if isinstance(visual_features, dict):
                        # Try common keys for V-JEPA
                        for key in ['x_norm_patchtokens', 'x', 'last_hidden_state', 'pooler_output']:
                            if key in visual_features:
                                visual_features = visual_features[key]
                                break
                        else:
                            # Fallback: use first value
                            visual_features = list(visual_features.values())[0]
                
                # Project visual features
                visual_embeddings = model.vision_proj(visual_features)
                
                # 2. Get query embeddings
                query_embeddings = model.token_embedding(input_ids)
                
                # 3. Encode captions (target)
                with torch.no_grad():
                    target_features = model.y_encoder.encode(
                        captions, 
                        convert_to_tensor=True, 
                        show_progress_bar=False, 
                        device=device
                    )
                
                # Project target features
                target_embeddings = model.y_encoder_proj(target_features)
                
                # 4. Forward through predictor
                pred_embeddings = model(visual_embeddings, query_embeddings, attention_mask)
                
                # 5. Compute loss
                loss = bidirectional_infonce_loss(
                    pred_embeddings, 
                    target_embeddings, 
                    temperature=model.config.temperature
                )
            
            # Backward pass
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            valid_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / valid_batches:.4f}"
            })
            
        except Exception as e:
            print(f"\n⚠️  Error in batch {batch_idx}: {e}")
            continue
    
    # Calculate average loss
    avg_loss = total_loss / valid_batches if valid_batches > 0 else float('inf')
    
    return {
        'loss': avg_loss,
        'num_batches': num_batches,
        'valid_batches': valid_batches
    }


def train_vljepa(
    model: VLJEPA,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 5,
    device: str = 'cuda',
    use_amp: bool = True,
    save_dir: str = 'checkpoints',
    log_interval: int = 10
):
    """
    Main training loop
    """
    model.to(device)
    
    # Setup mixed precision training
    scaler = GradScaler() if use_amp else None
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Training history
    history = {
        'epoch': [],
        'loss': [],
        'time': []
    }
    
    print("\n" + "="*60)
    print(f"Starting training for {num_epochs} epochs")
    print(f"Device: {device}")
    print(f"Mixed Precision: {use_amp}")
    print(f"Batches per epoch: {len(train_dataloader)}")
    print("="*60 + "\n")
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Train one epoch
        metrics = train_epoch(
            model, 
            train_dataloader, 
            optimizer,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
            epoch=epoch
        )
        
        epoch_time = time.time() - epoch_start
        
        # Log results
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Valid batches: {metrics['valid_batches']}/{metrics['num_batches']}")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Save history
        history['epoch'].append(epoch)
        history['loss'].append(metrics['loss'])
        history['time'].append(epoch_time)
        
        # Save checkpoint
        checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': metrics['loss'],
            'config': model.config
        }, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(save_dir, 'final_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model.config,
        'history': history
    }, final_path)
    print(f"\n✓ Final model saved: {final_path}")
    
    # Save training history
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Training history saved: {history_path}")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train VL-JEPA model')
    parser.add_argument('--video_root', type=str, default='assets', help='Root directory for videos')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Base learning rate')
    parser.add_argument('--vision_lr', type=float, default=None, help='Vision projection LR (if None, uses base LR)')
    parser.add_argument('--predictor_lr', type=float, default=None, help='Predictor LR (if None, uses base LR)')
    parser.add_argument('--y_encoder_lr', type=float, default=None, help='Y-encoder LR (if None, uses base LR * multiplier)')
    parser.add_argument('--y_encoder_lr_multiplier', type=float, default=0.05, help='Y-encoder LR multiplier')
    parser.add_argument('--max_samples', type=int, default=None, help='Limit dataset size')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--no_amp', action='store_true', help='Disable mixed precision training')
    
    args = parser.parse_args()
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. Training on CPU will be very slow.")
        device = 'cpu'
    else:
        device = 'cuda'
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Load dataset from local videos
    print("\nLoading local video dataset...")
    try:
        dataset = LocalVideoDataset(
            config=config,
            video_root=args.video_root,
            max_samples=args.max_samples
        )
        print(f"✓ Dataset loaded: {len(dataset)} video files")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0,  # Set to 0 for debugging, increase for production
        pin_memory=True if device == 'cuda' else False
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = VLJEPA(config)
    
    # Setup optimizer with proper parameter groups (different LR for each component)
    optimizer = AdamW([
        {'params': model.vision_proj.parameters(), 'lr': config.get_vision_proj_lr(), 'name': 'vision_proj'},
        {'params': model.predictor_layers.parameters(), 'lr': config.get_predictor_lr(), 'name': 'predictor_layers'},
        {'params': model.predictor_norm.parameters(), 'lr': config.get_predictor_lr(), 'name': 'predictor_norm'},
        {'params': model.predictor_proj.parameters(), 'lr': config.get_predictor_lr(), 'name': 'predictor_proj'},
        {'params': model.y_encoder.parameters(), 'lr': config.get_y_encoder_lr(), 'name': 'y_encoder'},
        {'params': model.y_encoder_proj.parameters(), 'lr': config.get_y_encoder_lr(), 'name': 'y_encoder_proj'},
    ])
    
    print(f"✓ Optimizer configured with differential learning rates:")
    print(f"  Vision Projection LR: {config.get_vision_proj_lr():.2e}")
    print(f"  Predictor LR: {config.get_predictor_lr():.2e}")
    print(f"  Y-Encoder LR: {config.get_y_encoder_lr():.2e}")
    print(f"  LR Ratio (Y-Encoder/Base): {config.get_y_encoder_lr()/config.learning_rate:.2f}x")
    
    # Train
    model, history = train_vljepa(
        model=model,
        train_dataloader=dataloader,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        device=device,
        use_amp=not args.no_amp,
        save_dir=args.save_dir
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == "__main__":
    # Check if running as script with args or as module
    if len(sys.argv) > 1:
        main()
    else:
        # Quick test/demo mode
        print("Running in demo mode (use command line args for full training)")
        print("Example: python train_fixed.py --video_root /path/to/videos --batch_size 4\n")
        
        config = VLJEPAConfig()
        
        if not DATASETS_AVAILABLE:
            print("❌ 'datasets' library required. Install with: pip install datasets")
            sys.exit(1)
        
        # Load small dataset from local videos
        print("Loading local video dataset...")
        dataset = LocalVideoDataset(
            config=config,
            video_root=r"C:\Users\bahaa\Desktop\Optimized-JEPA\assets",
            max_samples=5
        )
        
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        print("Initializing Model...")
        model = VLJEPA(config)
        
        # Setup optimizer with differential learning rates
        optimizer = AdamW([
            {'params': model.vision_proj.parameters(), 'lr': config.get_vision_proj_lr()},
            {'params': model.predictor_layers.parameters(), 'lr': config.get_predictor_lr()},
            {'params': model.predictor_norm.parameters(), 'lr': config.get_predictor_lr()},
            {'params': model.predictor_proj.parameters(), 'lr': config.get_predictor_lr()},
            {'params': model.y_encoder.parameters(), 'lr': config.get_y_encoder_lr()},
            {'params': model.y_encoder_proj.parameters(), 'lr': config.get_y_encoder_lr()},
        ])
        
        if torch.cuda.is_available():
            print("\n✓ Starting quick training test...")
            model, history = train_vljepa(
                model=model,
                train_dataloader=dataloader,
                optimizer=optimizer,
                num_epochs=1,
                device='cuda',
                use_amp=True,
                save_dir='checkpoints_demo'
            )
            print(f"\n✓ Demo complete! Average Loss: {history['loss'][0]:.4f}")
        else:
            print("⚠️  CUDA not available - skipping training demo")