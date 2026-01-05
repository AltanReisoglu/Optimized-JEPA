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
    learning_rate = 5e-5
    y_encoder_lr_multiplier = 0.05
    temperature = 0.07

class VLJEPA(nn.Module):
    def __init__(self, config: VLJEPAConfig):
        super().__init__()
        self.config = config
        self.processor = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_preprocessor')
        self.x_encoder = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_large')[0]
        for param in self.x_encoder.parameters():
            param.requires_grad = False
        self.x_encoder.eval()
        self.vision_proj = nn.Linear(
            config.vjepa_hidden_dim, 
            config.predictor_hidden_dim
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.predictor_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        predictor_config = AutoConfig.from_pretrained(config.predictor_id)
        self.rope = LlamaRotaryEmbedding(
            config=predictor_config
        )
        
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
        for layer in self.predictor_layers:
            if hasattr(layer.self_attn, 'is_causal'):
                layer.self_attn.is_causal = False
        del full_model
        torch.cuda.empty_cache()
        self.y_encoder = SentenceTransformer(config.y_encoder_id)
        self.predictor_proj = nn.Linear(
            config.predictor_hidden_dim,
            config.embedding_dim
        )
        self.y_encoder_proj = nn.Linear(
            config.y_encoder_hidden_dim,
            config.embedding_dim
        )

    def encode_video(self, video_path):
        if isinstance(video_path, str):
            video_path = [video_path]
        video_tensors = []
        device = next(self.parameters()).device
        for path in video_path:
            vr = VideoReader(path, ctx=cpu(0))
            num_frames = self.config.VIDEO_INPUT_FRAMES
            indices = np.linspace(0, len(vr) - 1, num_frames).astype(int)
            frames = vr.get_batch(indices).asnumpy()
            frames_tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2)
            processed_video = self.processor(frames_tensor)[0]
            video_tensors.append(processed_video)
        batch_video_tensor = torch.stack(video_tensors).to(device)
        with torch.no_grad():
            visual_features = self.x_encoder(batch_video_tensor)
        visual_embeddings = self.vision_proj(visual_features)
        return visual_embeddings
    
    def encode_query(self, query_text):
        if isinstance(query_text, str):
            query_text = [query_text]
        tokens = self.tokenizer(
            query_text,
            max_length=self.config.max_query_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        device = next(self.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}
        query_embeddings = self.token_embedding(tokens['input_ids'])
        attention_mask = tokens['attention_mask']
        return query_embeddings, attention_mask
    
    def encode_target(self, target_text):
        if isinstance(target_text, str):
            target_text = [target_text]
        with torch.no_grad():
            y_features = self.y_encoder.encode(
                target_text,
                convert_to_tensor=True,
                show_progress_bar=False
            )
        target_embeddings = self.y_encoder_proj(y_features)
        target_embeddings = F.normalize(target_embeddings, dim=-1)
        return target_embeddings
    
    def forward(self, video_path, query_text, target_text=None):
        visual_embeddings = self.encode_video(video_path)
        B, N_visual, D = visual_embeddings.shape
        query_embeddings, query_attention_mask = self.encode_query(query_text)
        
        predictor_dtype = self.predictor_layers[0].self_attn.q_proj.weight.dtype
        visual_embeddings = visual_embeddings.to(dtype=predictor_dtype)
        query_embeddings = query_embeddings.to(dtype=predictor_dtype)
        
        combined = torch.cat([visual_embeddings, query_embeddings], dim=1)
        seq_length = combined.size(1)
        device = combined.device
        
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(B, -1)
        
        visual_mask = torch.ones(B, N_visual, dtype=torch.bool, device=device)

        query_attention_mask_bool = query_attention_mask.bool()
        attention_mask = torch.cat([visual_mask, query_attention_mask_bool], dim=1)
        

        hidden_states = combined
        

        cos, sin = self.rope(hidden_states, position_ids)
        position_embeddings = (cos, sin)

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
        
        query_output = hidden_states[:, -self.config.max_query_length:, :]
        mask = query_attention_mask_bool.unsqueeze(-1).to(device=device, dtype=hidden_states.dtype)
        pooled = (query_output * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        pooled = pooled.to(dtype=self.predictor_proj.weight.dtype)
        predicted_embedding = self.predictor_proj(pooled)
        predicted_embedding = F.normalize(predicted_embedding, dim=-1)
        
        if target_text is not None:
            target_embedding = self.encode_target(target_text)
            return predicted_embedding, target_embedding
        return predicted_embedding

def bidirectional_infonce_loss(pred_embeddings, target_embeddings, temperature=0.07):
    logits = torch.matmul(pred_embeddings, target_embeddings.t()) / temperature
    labels = torch.arange(pred_embeddings.size(0), device=pred_embeddings.device)
    loss1 = F.cross_entropy(logits, labels)
    loss2 = F.cross_entropy(logits.t(), labels)
    loss = (loss1 + loss2) / 2
    return loss

def train_vljepa(model, dataloader, optimizer, device='cuda'):
    model.train()
    model.x_encoder.eval()
    total_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        video_paths = batch['video_path']
        queries = batch['query']
        targets = batch['target']
        optimizer.zero_grad()
        pred_emb, target_emb = model(video_paths, queries, targets)
        loss = bidirectional_infonce_loss(
            pred_emb, 
            target_emb, 
            temperature=model.config.temperature
        )
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    avg_loss = total_loss / len(dataloader)
    return avg_loss

@torch.no_grad()
def inference_classification(model, video_path, class_names, query="What activity is shown?"):
    model.eval()
    predicted_embedding = model(video_path, query)
    class_embeddings = model.encode_target(class_names)
    similarities = torch.matmul(predicted_embedding, class_embeddings.t())
    probs = F.softmax(similarities / model.config.temperature, dim=-1)
    probs = probs.squeeze(0).cpu().numpy()
    predicted_idx = similarities.argmax(dim=-1).item()
    predicted_class = class_names[predicted_idx]
    confidences = {name: float(prob) for name, prob in zip(class_names, probs)}
    return predicted_class, confidences

@torch.no_grad()
def inference_vqa(model, video_path, question, answer_candidates):
    return inference_classification(model, video_path, answer_candidates, query=question)

@torch.no_grad()
def inference_retrieval(model, video_database, text_query, top_k=5):
    model.eval()
    video_embeddings = []
    print(f"Encoding {len(video_database)} videos...")
    for video_path in video_database:
        emb = model(video_path, query="Caption this video")
        video_embeddings.append(emb)
    video_embeddings = torch.cat(video_embeddings, dim=0)
    text_embedding = model.encode_target(text_query)
    similarities = torch.matmul(text_embedding, video_embeddings.t())
    top_scores, top_indices = similarities.topk(k=min(top_k, len(video_database)), dim=-1)
    top_scores = top_scores.squeeze(0).cpu().numpy()
    top_indices = top_indices.squeeze(0).cpu().numpy()
    top_videos = [
        (video_database[idx], float(score))
        for idx, score in zip(top_indices, top_scores)
    ]
    return top_videos

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
        {'params': model.vision_proj.parameters(), 'lr': config.learning_rate},
        {'params': model.predictor_layers.parameters(), 'lr': config.learning_rate},
        {'params': model.predictor_norm.parameters(), 'lr': config.learning_rate},
        {'params': model.predictor_proj.parameters(), 'lr': config.learning_rate},
        {'params': model.y_encoder.parameters(), 'lr': config.learning_rate * config.y_encoder_lr_multiplier},
        {'params': model.y_encoder_proj.parameters(), 'lr': config.learning_rate * config.y_encoder_lr_multiplier},
    ])
    print(f"✓ Optimizer configured")
    print(f"  Base LR: {config.learning_rate}")
    print(f"  Y-Encoder LR: {config.learning_rate * config.y_encoder_lr_multiplier}")
    print("\n" + "="*60)
    print("VL-JEPA Ready for Training!")
    print("="*60)
