import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import dataclasses
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from sentence_transformers import SentenceTransformer
import numpy as np
from decord import VideoReader, cpu
from typing import List, Dict, Tuple
import os

@dataclasses.dataclass
class VLJEPAConfig:
    VIDEO_INPUT_SIZE = (224, 224)
    VIDEO_INPUT_FRAMES = 16
    vjepa_hidden_dim = 1024
    predictor_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    predictor_hidden_dim = 2048
    predictor_num_layers = 22
    predictor_start_layer = 10
    predictor_end_layer = 18
    y_encoder_id = "sentence-transformers/all-MiniLM-L6-v2"
    y_encoder_hidden_dim = 384
    embedding_dim = 2048
    max_query_length = 512
    
    # Learning rates - differential per component
    learning_rate = 5e-5  # Base LR for vision_proj, predictor_layers, predictor_norm, predictor_proj
    y_encoder_lr_multiplier = 0.05  # Y-encoder uses: learning_rate * y_encoder_lr_multiplier
    
    # Alternative: specify each component's LR individually
    vision_proj_lr = None  # If None, uses learning_rate
    predictor_lr = None  # If None, uses learning_rate
    y_encoder_lr = None  # If None, uses learning_rate * y_encoder_lr_multiplier
    
    temperature = 0.07
    
    def get_vision_proj_lr(self):
        return self.vision_proj_lr if self.vision_proj_lr is not None else self.learning_rate
    
    def get_predictor_lr(self):
        return self.predictor_lr if self.predictor_lr is not None else self.learning_rate
    
    def get_y_encoder_lr(self):
        if self.y_encoder_lr is not None:
            return self.y_encoder_lr
        return self.learning_rate * self.y_encoder_lr_multiplier

class VLJEPA(nn.Module):
    def __init__(self, config: VLJEPAConfig):
        super().__init__()
        self.config = config
        
        # Video processor ve encoder
        print("Loading V-JEPA preprocessor...")
        self.processor = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_preprocessor')
        print("Loading V-JEPA encoder...")
        self.x_encoder = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_large')[0]
        
        # Freeze vision encoder
        for param in self.x_encoder.parameters():
            param.requires_grad = False
        self.x_encoder.eval()
        
        # Vision projection
        self.vision_proj = nn.Linear(
            config.vjepa_hidden_dim, 
            config.predictor_hidden_dim
        )
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.predictor_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # RoPE
        predictor_config = AutoConfig.from_pretrained(config.predictor_id)
        self.rope = LlamaRotaryEmbedding(config=predictor_config)
        
        # Predictor layers
        print(f"Loading predictor from {config.predictor_id}...")
        full_model = AutoModelForCausalLM.from_pretrained(
            config.predictor_id,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        self.token_embedding = full_model.model.embed_tokens
        self.predictor_layers = nn.ModuleList([
            full_model.model.layers[i] 
            for i in range(config.predictor_start_layer, config.predictor_end_layer)
        ])
        self.predictor_norm = full_model.model.norm
        
        # Disable causal masking for cross-attention style processing
        for layer in self.predictor_layers:
            if hasattr(layer.self_attn, 'is_causal'):
                layer.self_attn.is_causal = False
        
        del full_model
        torch.cuda.empty_cache()
        
        # Y encoder (text)
        print(f"Loading text encoder {config.y_encoder_id}...")
        self.y_encoder = SentenceTransformer(config.y_encoder_id)
        
        # Projections to common space
        self.predictor_proj = nn.Linear(
            config.predictor_hidden_dim,
            config.embedding_dim
        )
        self.y_encoder_proj = nn.Linear(
            config.y_encoder_hidden_dim,
            config.embedding_dim
        )
        
        print("Model initialized successfully!")

    def process_video(self, video_path: str) -> torch.Tensor:
        """Process video file and return tensor for encoding"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            
            # Sample frames uniformly
            indices = np.linspace(0, total_frames - 1, self.config.VIDEO_INPUT_FRAMES).astype(int)
            frames = vr.get_batch(indices).asnumpy()
            
            # Convert to tensor: (T, H, W, C) -> (T, C, H, W)
            # Normalize to [0, 1] range
            frames_tensor = torch.from_numpy(frames).float() / 255.0
            frames_tensor = frames_tensor.permute(0, 3, 1, 2)
            
            # Process with V-JEPA preprocessor
            # Ensure input is (C, T, H, W) for standard transforms if needed, 
            # though VJEPA preprocessor usually handles (T, C, H, W) or (C, T, H, W).
            # We'll stick to what we fixed in train.py for consistency: input as (C, T, H, W)
            frames_input = frames_tensor.permute(1, 0, 2, 3) 
            processed = self.processor(frames_input)
            
            # Handle different return types
            if isinstance(processed, list):
                processed = processed[0]
            elif isinstance(processed, dict):
                # Try common keys
                if 'x' in processed:
                    processed = processed['x']
                else:
                    processed = list(processed.values())[0]
            
            return processed
            
        except Exception as e:
            raise RuntimeError(f"Error processing video {video_path}: {str(e)}")

    def encode_video(self, video_path: str) -> torch.Tensor:
        """Encode video to visual embeddings"""
        pixel_values = self.process_video(video_path)
        
        # Add batch dimension if needed
        if pixel_values.dim() == 4:  # (C, T, H, W)
            pixel_values = pixel_values.unsqueeze(0)
        
        device = next(self.parameters()).device
        pixel_values = pixel_values.to(device)
        
        with torch.no_grad():
            visual_features = self.x_encoder(pixel_values)
            
            # Handle dict output
            if isinstance(visual_features, dict):
                # Common keys: 'x_norm_patchtokens', 'x', 'last_hidden_state'
                for key in ['x_norm_patchtokens', 'x', 'last_hidden_state']:
                    if key in visual_features:
                        visual_features = visual_features[key]
                        break
                else:
                    # Fallback to first value
                    visual_features = list(visual_features.values())[0]
        
        visual_embeddings = self.vision_proj(visual_features)
        return visual_embeddings

    def forward(self, visual_embeddings, query_embeddings, query_attention_mask):
        """
        Args:
            visual_embeddings: (B, N_visual, D)
            query_embeddings: (B, N_query, D)
            query_attention_mask: (B, N_query) - 1 for real tokens, 0 for padding
        """
        B, N_visual, D = visual_embeddings.shape
        N_query = query_embeddings.size(1)
        
        # Type conversion
        predictor_dtype = self.predictor_layers[0].self_attn.q_proj.weight.dtype
        visual_embeddings = visual_embeddings.to(dtype=predictor_dtype)
        query_embeddings = query_embeddings.to(dtype=predictor_dtype)
        
        # Combine visual and text embeddings
        combined = torch.cat([visual_embeddings, query_embeddings], dim=1)
        seq_length = combined.size(1)
        device = combined.device
        
        # Position IDs
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(B, -1)
        
        # Create proper attention mask (4D: batch, 1, seq_len, seq_len)
        # Visual tokens attend to everything, query tokens attend based on mask
        visual_mask = torch.ones(B, N_visual, dtype=torch.bool, device=device)
        query_mask = query_attention_mask.bool()
        mask_1d = torch.cat([visual_mask, query_mask], dim=1)  # (B, seq_len)
        
        # Convert to 4D causal mask format
        # Shape: (B, 1, seq_len, seq_len)
        
        # Vectorized Attention Mask Creation (REPLACES SLOW LOOPS)
        # mask_1d: (B, seq_len)
        # We want to allow attention only if BOTH position i and j are valid (1)
        
        # Expand for broadcasting:
        # (B, 1, seq_len, 1)
        mask_rows = mask_1d.unsqueeze(1).unsqueeze(3) 
        # (B, 1, 1, seq_len)
        mask_cols = mask_1d.unsqueeze(1).unsqueeze(2)
        
        # Logical AND: Both must be valid to attend
        # Result: (B, 1, seq_len, seq_len)
        attention_allowed = mask_rows & mask_cols
        
        # Create mask tensor initialized to -inf
        attention_mask = torch.full((B, 1, seq_length, seq_length), float('-inf'), device=device, dtype=predictor_dtype)
        
        # Fill valid positions with 0.0
        attention_mask = attention_mask.masked_fill(attention_allowed, 0.0)
        
        # -----------------------------------------------------------------------
        # OLD SLOW CODE REMOVED (Nested loops caused the freeze)
        # -----------------------------------------------------------------------
        
        hidden_states = combined
        
        # RoPE embeddings
        cos, sin = self.rope(hidden_states, position_ids)
        position_embeddings = (cos, sin)
        
        # Pass through predictor layers
        for layer in self.predictor_layers:
            layer_output = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False,
                output_attentions=False,
            )
            if isinstance(layer_output, tuple):
                hidden_states = layer_output[0]
            else:
                hidden_states = layer_output
        
        hidden_states = self.predictor_norm(hidden_states)
        
        # Pool query outputs (only valid tokens)
        query_output = hidden_states[:, N_visual:, :]  # Get query part
        query_mask_expanded = query_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        
        # Average pooling over valid tokens
        pooled = (query_output * query_mask_expanded).sum(dim=1) / (query_mask_expanded.sum(dim=1) + 1e-8)
        
        # Project to embedding space
        pooled = pooled.to(dtype=self.predictor_proj.weight.dtype)
        predicted_embedding = self.predictor_proj(pooled)
        predicted_embedding = F.normalize(predicted_embedding, dim=-1)
        
        return predicted_embedding

    @torch.no_grad()
    def get_text_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Encode texts to embeddings"""
        device = next(self.parameters()).device
        
        # Encode with sentence transformer
        text_features = self.y_encoder.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            device=device
        )
        
        # Project to common space
        text_embeddings = self.y_encoder_proj(text_features)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        
        return text_embeddings

    @torch.no_grad()
    def predict(self, video_path: str, query: str) -> torch.Tensor:
        """
        Full inference pipeline: video + query -> embedding
        """
        device = next(self.parameters()).device
        
        # 1. Encode video
        visual_embeddings = self.encode_video(video_path)
        
        # 2. Tokenize query
        tokens = self.tokenizer(
            query,
            max_length=self.config.max_query_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)
        
        # 3. Get query embeddings
        query_embeddings = self.token_embedding(input_ids)
        
        # 4. Forward pass
        predicted_embedding = self.forward(visual_embeddings, query_embeddings, attention_mask)
        
        return predicted_embedding


def inference_classification(
    model: VLJEPA,
    video_path: str,
    class_names: List[str],
    query: str = "What is shown in this video?"
) -> Tuple[str, Dict[str, float]]:
    """
    Zero-shot classification
    
    Args:
        model: VLJEPA model
        video_path: Path to video file
        class_names: List of possible class names
        query: Question to ask about the video
    
    Returns:
        predicted_class: Most likely class name
        confidences: Dictionary of class -> confidence score
    """
    model.eval()
    
    # Get video+query embedding
    video_embedding = model.predict(video_path, query)
    
    # Get class embeddings
    class_embeddings = model.get_text_embeddings(class_names)
    
    # Compute similarities (cosine similarity via dot product of normalized vectors)
    similarities = torch.matmul(video_embedding, class_embeddings.t())
    
    # Convert to probabilities with temperature
    probs = F.softmax(similarities / model.config.temperature, dim=-1)
    
    # Get confidences
    confidences = {
        name: prob.item() 
        for name, prob in zip(class_names, probs[0])
    }
    
    # Get prediction
    predicted_idx = probs.argmax(dim=-1).item()
    predicted_class = class_names[predicted_idx]
    
    return predicted_class, confidences


def inference_vqa(
    model: VLJEPA,
    video_path: str,
    question: str,
    answers: List[str]
) -> Tuple[str, Dict[str, float]]:
    """
    Visual Question Answering
    
    Args:
        model: VLJEPA model
        video_path: Path to video file
        question: Question to ask
        answers: List of possible answers
    
    Returns:
        predicted_answer: Most likely answer
        confidences: Dictionary of answer -> confidence score
    """
    model.eval()
    
    # Get video+question embedding
    video_embedding = model.predict(video_path, question)
    
    # Get answer embeddings
    answer_embeddings = model.get_text_embeddings(answers)
    
    # Compute similarities
    similarities = torch.matmul(video_embedding, answer_embeddings.t())
    
    # Convert to probabilities
    probs = F.softmax(similarities / model.config.temperature, dim=-1)
    
    # Get confidences
    confidences = {
        ans: prob.item() 
        for ans, prob in zip(answers, probs[0])
    }
    
    # Get prediction
    predicted_idx = probs.argmax(dim=-1).item()
    predicted_answer = answers[predicted_idx]
    
    return predicted_answer, confidences


if __name__ == "__main__":
    print("="*60)
    print("Initializing VL-JEPA...")
    print("="*60)
    
    config = VLJEPAConfig()
    model = VLJEPA(config)
    
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"\n✓ Model moved to GPU")
    
    print("="*60)
    print("\n" + "="*60)
    print("TEST 1: Zero-Shot Classification")
    print("="*60)
    
    video_path = r"C:\Users\bahaa\Desktop\Optimized-JEPA\assets\vecteezy_a-cyclist-rides-on-a-bike-path-in-the-city_27944714.mov"
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"⚠️  Video not found: {video_path}")
        print("Please update the video_path variable with a valid video file")
    else:
        class_names = [
            "cycling",
            "running",
            "swimming",
            "cooking",
            "reading"
        ]
        
        predicted_class, confidences = inference_classification(
            model, 
            video_path, 
            class_names,
            query="What activity is shown in this video?"
        )
        
        print(f"\nPredicted Activity: {predicted_class}")
        print("\nConfidences:")
        for name, conf in sorted(confidences.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name}: {conf:.4f}")
        
        print("\n" + "="*60)
        print("TEST 2: Visual Question Answering")
        print("="*60)
        
        question = "What is the person doing?"
        answers = [
            "riding a bicycle",
            "driving a car",
            "walking",
            "sitting"
        ]
        
        predicted_answer, vqa_confidences = inference_vqa(
            model,
            video_path,
            question,
            answers
        )
        
        print(f"\nQuestion: {question}")
        print(f"Answer: {predicted_answer}")
        print("\nAll answers:")
        for ans, conf in sorted(vqa_confidences.items(), key=lambda x: x[1], reverse=True):
            print(f"  {ans}: {conf:.4f}")
    
    print("\n" + "="*60)
    print("Setting up optimizer...")
    print("="*60)
    
    optimizer = optim.AdamW([
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
    
    print("\n" + "="*60)
    print("VL-JEPA Ready for Training!")
    print("="*60)