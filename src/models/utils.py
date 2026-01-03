import os
import torch
import cv2
import numpy as np  


import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange, repeat


def compute_plucker_coordinates(z_left, z_right):
    """
    İki vektör çiftinden Plücker koordinatlarını hesapla.
    
    Args:
        z_left: (batch, seq_len, r) - Sol vektörler
        z_right: (batch, seq_len, r) - Sağ vektörler
    
    Returns:
        p: (batch, seq_len, r*(r-1)/2) - Plücker koordinatları
    """
    batch_size, seq_len, r = z_left.shape
    
    # Plücker koordinatları: p_ij = z_left_i * z_right_j - z_left_j * z_right_i
    # Tüm i < j çiftleri için
    
    plucker_coords = []
    
    for i in range(r):
        for j in range(i + 1, r):
            # p_ij hesapla
            p_ij = z_left[:, :, i] * z_right[:, :, j] - z_left[:, :, j] * z_right[:, :, i]
            plucker_coords.append(p_ij)
    
    # Stack: (batch, seq_len, r*(r-1)/2)
    p = torch.stack(plucker_coords, dim=-1)
    
    return p
def normalize_pdelta(p_delta):
    """
    Args:
        p_delta: (batch, seq_len, r*(r-1)/2) - Plücker koordinatları
    Returns:
        p_delta: (batch, seq_len, r*(r-1)/2) - Normalleştirilmiş Plücker koordinatları
    """
    max_norm = torch.max(torch.norm(p_delta, dim=-1, keepdim=True))
    return p_delta / max_norm



def compute_plucker_coordinates_efficient(z_left, z_right):
    """
    Vektörize edilmiş, daha hızlı versiyon.
    
    Args:
        z_left: (batch, seq_len, r)
        z_right: (batch, seq_len, r)
    
    Returns:
        p: (batch, seq_len, r*(r-1)/2)
    """
    batch_size, seq_len, r = z_left.shape
    
    # z_left ve z_right'ı genişlet: (batch, seq_len, r, 1) ve (batch, seq_len, 1, r)
    z_left_expanded = z_left.unsqueeze(-1)  # (batch, seq_len, r, 1)
    z_right_expanded = z_right.unsqueeze(-2)  # (batch, seq_len, 1, r)
    
    # Dış çarpım: (batch, seq_len, r, r)
    outer_product = z_left_expanded * z_right_expanded
    
    # Antisimetrik kısım: A - A^T
    antisymmetric = outer_product - outer_product.transpose(-2, -1)
    
    # Üst üçgen kısmı al (i < j)
    indices = torch.triu_indices(r, r, offset=1)
    p = antisymmetric[:, :, indices[0], indices[1]]
    
    return p


class RMSNorm(nn.Module):
    def __init__(
        self, num_features: int, eps: float = 1e-05, device: torch.device | None = None
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = torch.nn.Parameter(
            torch.ones(num_features, device=device, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.num_features
        t, dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return (t * self.scale).to(dtype)
    
def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)

import math
class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        use_xpos = False,
        scale_base = 512,
        interpolation_factor = 1.,
        base = 10000,
        base_rescale_factor = 1.
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        base *= base_rescale_factor ** (dim / (dim - 2))

        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        assert interpolation_factor >= 1.
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.register_buffer('scale', None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        self.register_buffer('scale', scale)

    def forward(self, seq_len):
        device = self.inv_freq.device
        t = torch.arange(seq_len, device = device).type_as(self.inv_freq)

        t = t / self.interpolation_factor

        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)

        if self.scale is None:
            return freqs, 1.

        power = (torch.arange(seq_len, device = device) - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1)

        return freqs, scale


def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb(t, freqs, scale = 1):
    rot_dim, seq_len = freqs.shape[-1], t.shape[-2]
    freqs = freqs[-seq_len:, :]

    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, 'b n d -> b 1 n d')

    # partial rotary embeddings, Wang et al. GPT-J
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t, t_unrotated), dim = -1)


class DoRALinear(nn.Module):
    @staticmethod
    def from_linear(linear, r, alpha, scale, dropout):
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims *= 32 // linear.bits
        lora_lin = DoRALinear(input_dims=input_dims, output_dims=output_dims, r=r, alpha=alpha, scale=scale, dropout=dropout)
        lora_lin.linear = linear
        return lora_lin

    def __init__(self, input_dims, output_dims, r, alpha, scale, dropout, bias=False):
        super().__init__()
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)
        self.dropout = nn.Dropout(p=dropout)
        self.scale = scale * (alpha / r)
        init_scale = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(low=-init_scale, high=init_scale, shape=(input_dims, r))
        self.lora_b = mx.zeros(shape=(r, output_dims))
        self.m = mx.linalg.norm(self._dequantized_weight().astype(mx.float32), axis=1)

    def _dequantized_weight(self):
        weight = self.linear.weight
        if isinstance(self.linear, nn.QuantizedLinear):
            weight = mx.dequantize(weight, self.linear.scales, self.linear.biases, self.linear.group_size, self.linear.bits)
        return weight

    def __call__(self, x):
        bias = self.linear.bias if "bias" in self.linear else 0
        y = self.linear(x)
        y = y - bias
        z = (self.dropout(x) @ self.lora_a) @ self.lora_b
        z = y + (self.scale * z)
        w = self._dequantized_weight()
        adapted = w + (self.scale * self.lora_b.T) @ self.lora_a.T
        denom = mx.linalg.norm(adapted, axis=1) + 1e-6
        z = (self.m / denom) * z
        z = z + bias
        return z

def to_dora(layers, targets=None, rank=8, scale=0.1):
    _targets = ['o_proj', 'down_proj'] if targets is None else targets
    for l in layers:
        loralized = [(k, DoRALinear.from_linear(m, r=rank, alpha=rank, scale=scale, dropout=0.0)) for k, m in l.named_modules() if any(k.endswith(_t) for _t in _targets)]
        l.update_modules(tree_unflatten(loralized))
class BidirectionalAntiAttention(nn.Module):
    """
    Bidirectional Anti-Attention: Hem geçmiş hem gelecek tokenları kullanarak
    Grassmann manifold üzerinde özellik çıkarımı yapar.
    """
    def __init__(self, r, d, window_offsets, dropout=0.1):
        super().__init__()
        self.window_offsets = window_offsets
        self.r_to_d = nn.Linear(r, d)
        self.d_to_r = nn.Linear(d, r)
        self.plucker_shape = r * (r - 1) // 2
        self.plucker_to_d = nn.Linear(self.plucker_shape, d)
        self.d_to_plucker = nn.Linear(d, self.plucker_shape)
        
        # Forward ve backward için ayrı projeksiyonlar
        self.plucker_projection_forward = nn.Linear(self.plucker_shape, d)
        self.plucker_projection_backward = nn.Linear(self.plucker_shape, d)
        
        # Forward, backward ve original için gate
        self.gate = nn.Linear(3 * d, d)
        self.layer_norm = RMSNorm(d)
        self.dropout = nn.Dropout(dropout)

    def _compute_directional_features(self, z, direction='forward'):
        """
        Tek yönlü Grassmann özelliklerini hesapla.
        
        Args:
            z: (batch, seq_len, r) - Reduced representations
            direction: 'forward' veya 'backward'
        
        Returns:
            g: (batch, seq_len, d) - Grassmann features
            valid_counts: (batch, seq_len) - Valid position counts
        """
        b, l, r = z.shape
        d = self.plucker_projection_forward.out_features
        
        grassmann_features = []
        valid_counts = torch.zeros(b, l, device=z.device)
        
        projection = self.plucker_projection_forward if direction == 'forward' else self.plucker_projection_backward
        
        for delta in self.window_offsets:
            if delta >= l:
                continue
            
            if direction == 'forward':
                # Forward: şimdiki pozisyon -> gelecek pozisyon
                z_left = z[:, :-delta, :]   # (batch, seq_len-delta, r)
                z_right = z[:, delta:, :]   # (batch, seq_len-delta, r)
            else:
                # Backward: şimdiki pozisyon -> geçmiş pozisyon
                z_left = z[:, delta:, :]    # (batch, seq_len-delta, r)
                z_right = z[:, :-delta, :]  # (batch, seq_len-delta, r)
            
            # Plücker koordinatlarını hesapla
            p = compute_plucker_coordinates_efficient(z_left, z_right)
            
            # Normalize
            p_norm = torch.norm(p, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
            p_normalized = p / p_norm
            
            # Project to d dimensions
            g_delta = projection(p_normalized)
            
            # Padding: forward için sona, backward için başa
            if direction == 'forward':
                g_delta_padded = F.pad(g_delta, (0, 0, 0, delta), value=0.0)
                valid_mask_padded = F.pad(torch.ones(b, l - delta, device=z.device), (0, delta), value=0.0)
            else:
                g_delta_padded = F.pad(g_delta, (0, 0, delta, 0), value=0.0)
                valid_mask_padded = F.pad(torch.ones(b, l - delta, device=z.device), (delta, 0), value=0.0)
            
            grassmann_features.append(g_delta_padded)
            valid_counts += valid_mask_padded
        
        if len(grassmann_features) > 0:
            grassmann_features_stacked = torch.stack(grassmann_features, dim=0)
            grassmann_features_sum = grassmann_features_stacked.sum(dim=0)
            valid_counts = valid_counts.unsqueeze(-1).clamp(min=1.0)
            g = grassmann_features_sum / valid_counts
        else:
            g = torch.zeros(b, l, d, device=z.device)
        
        return g

    def forward(self, x):
        """
        Bidirectional forward pass.
        
        Args:
            x: (batch, seq_len, d) - Input embeddings
        
        Returns:
            h_mixed: (batch, seq_len, d) - Mixed output
        """
        b, l, d = x.shape
        
        # Reduce to r dimensions
        z = self.d_to_r(x)
        
        # Forward features (looking at future tokens)
        g_forward = self._compute_directional_features(z, direction='forward')
        
        # Backward features (looking at past tokens)
        g_backward = self._compute_directional_features(z, direction='backward')
        
        # Combine forward and backward with original
        concat = torch.cat([x, g_forward, g_backward], dim=-1)  # (batch, seq_len, 3*d)
        
        # Gated combination
        alpha = torch.sigmoid(self.gate(concat))  # (batch, seq_len, d)
        
        # Mix: weighted combination of original and bidirectional features
        g_combined = (g_forward + g_backward) / 2.0
        h_mixed = alpha * x + (1 - alpha) * g_combined
        
        # Normalize and dropout
        h_mixed = self.layer_norm(h_mixed)
        h_mixed = self.dropout(h_mixed)
        
        return h_mixed


# Backward compatibility alias
AntiAttention = BidirectionalAntiAttention


class FFN(nn.Module):

    def __init__(self, dimension,coeff=1):
        super().__init__()
        self.linear_1 = nn.Linear(dimension,dimension*coeff)
        self.linear_2 = nn.Linear(dimension*coeff,dimension)

    def forward(self, x):
        output = self.linear_1(x)
        swish = output * torch.sigmoid(output)
        
        swiglu = self.linear_2(swish)

        return swiglu 

class ResidualBlock(nn.Module):
    def __init__(self, dimension,reduction,coeffs):
        super().__init__()
        self.ffn = FFN(dimension,4)
        self.antia = AntiAttention(r=reduction, d=dimension, window_offsets=coeffs)
        self.layer_norm = RMSNorm(dimension)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        residue=x
        x = self.antia(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.ffn(x)
        x = residue + x

        return x
        



class GrassmannEncoder(nn.Module):
    """
    Grassmann manifold tabanlı encoder - hem görsel hem metin için kullanılabilir.
    Bidirectional özellikleri kullanarak zengin temsiller öğrenir.
    """
    
    def __init__(
        self,
        d_model=256,
        r=32,
        n_layers=6,
        window_offsets=[1, 2, 4, 8, 12, 16],
        max_seq_len=512,
        dropout=0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.positional_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            ResidualBlock(d_model, r, window_offsets)
            for _ in range(n_layers)
        ])
        
        self.final_layer_norm = RMSNorm(d_model)
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, add_pos_emb=True):
        """
        Args:
            x: (batch, seq_len, d_model) - Input embeddings
            add_pos_emb: Positional embedding eklensin mi
        
        Returns:
            hidden_states: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        if add_pos_emb:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            pos_embeds = self.positional_embedding(positions)
            x = x + pos_embeds
        
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.final_layer_norm(x)
        return x


# ============================================================================
# VL-JEPA Architecture Components
# ============================================================================

class XEncoder(nn.Module):
    """
    X-Encoder (Vision Encoder): Görsel girdileri kompakt visual embedding'lere sıkıştırır.
    V-JEPA 2 tarzı - yüksek hacimli görsel girdileri "visual tokens"a dönüştürür.
    
    Xv → Sv (Visual Embeddings)
    """
    
    def __init__(
        self,
        input_dim=768,          # Vision backbone output dim (ViT, CLIP, etc.)
        d_model=256,
        r=32,
        n_layers=4,
        window_offsets=[1, 2, 4, 8],
        max_seq_len=256,        # Max visual tokens
        dropout=0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Vision input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Grassmann-based encoder
        self.encoder = GrassmannEncoder(
            d_model=d_model,
            r=r,
            n_layers=n_layers,
            window_offsets=window_offsets,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        # Pooling for global representation
        self.pool_projection = nn.Linear(d_model, d_model)
        
    def forward(self, visual_features):
        """
        Args:
            visual_features: (batch, num_patches, input_dim) - Vision backbone çıktısı
        
        Returns:
            Sv: (batch, num_patches, d_model) - Visual embeddings
            Sv_pooled: (batch, d_model) - Global visual representation
        """
        # Project to model dimension
        x = self.input_projection(visual_features)
        
        # Encode with Grassmann layers
        Sv = self.encoder(x)
        
        # Global pooling (mean pooling)
        Sv_pooled = Sv.mean(dim=1)
        Sv_pooled = self.pool_projection(Sv_pooled)
        
        return Sv, Sv_pooled


class YEncoder(nn.Module):
    """
    Y-Encoder (Target Text Encoder): Metin hedefini sürekli latent uzaya embed eder.
    Hedef embedding, görev-dışı bilgiyi soyutlaştırması beklenir.
    
    Y → Sy (Target Embedding)
    """
    
    def __init__(
        self,
        vocab_size=32000,
        d_model=256,
        r=32,
        n_layers=4,
        window_offsets=[1, 2, 4, 8],
        max_seq_len=512,
        dropout=0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Grassmann-based encoder
        self.encoder = GrassmannEncoder(
            d_model=d_model,
            r=r,
            n_layers=n_layers,
            window_offsets=window_offsets,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        # Output projection for target embedding
        self.output_projection = nn.Linear(d_model, d_model)
        
    def forward(self, target_ids):
        """
        Args:
            target_ids: (batch, seq_len) - Target token IDs
        
        Returns:
            Sy: (batch, d_model) - Target embedding (pooled)
            Sy_seq: (batch, seq_len, d_model) - Sequence-level embeddings
        """
        # Token embeddings
        x = self.token_embedding(target_ids)
        
        # Encode
        Sy_seq = self.encoder(x)
        
        # Global pooling for target embedding
        Sy = Sy_seq.mean(dim=1)
        Sy = self.output_projection(Sy)
        
        return Sy, Sy_seq


class Predictor(nn.Module):
    """
    Predictor: Visual embeddings + Text query → Target embedding tahmini.
    VL-JEPA'nın çekirdek bileşeni - görsel embedding'leri metin sorgusu koşulunda
    hedef embedding tahminine dönüştürür.
    
    (Sv, Xq) → Ŝy
    """
    
    def __init__(
        self,
        d_model=256,
        r=32,
        n_layers=5,                 # Llama 5 layers gibi
        window_offsets=[1, 2, 4, 8, 12, 16],
        max_seq_len=768,            # visual + query tokens
        dropout=0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Visual projection
        self.visual_projection = nn.Linear(d_model, d_model)
        
        # Query projection  
        self.query_projection = nn.Linear(d_model, d_model)
        
        # Modality type embeddings
        self.visual_type_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.query_type_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Cross-modal Grassmann encoder
        self.encoder = GrassmannEncoder(
            d_model=d_model,
            r=r,
            n_layers=n_layers,
            window_offsets=window_offsets,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        # Output projection for predicted target embedding
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        
    def forward(self, Sv, Sq):
        """
        Args:
            Sv: (batch, num_visual_tokens, d_model) - Visual embeddings from X-Encoder
            Sq: (batch, num_query_tokens, d_model) - Query embeddings
        
        Returns:
            Sy_hat: (batch, d_model) - Predicted target embedding
        """
        batch_size = Sv.shape[0]
        
        # Project and add type embeddings
        Sv_proj = self.visual_projection(Sv) + self.visual_type_embedding
        Sq_proj = self.query_projection(Sq) + self.query_type_embedding
        
        # Concatenate visual and query tokens
        # [Visual Tokens] [Query Tokens]
        combined = torch.cat([Sv_proj, Sq_proj], dim=1)
        
        # Encode with Grassmann layers (bidirectional)
        encoded = self.encoder(combined, add_pos_emb=True)
        
        # Pool to get prediction
        # Use query-aware pooling: focus on query region
        num_visual = Sv.shape[1]
        query_features = encoded[:, num_visual:, :]  # Query part
        
        # Weighted pooling based on visual-query interaction
        Sy_hat = query_features.mean(dim=1)
        Sy_hat = self.output_projection(Sy_hat)
        
        return Sy_hat


class VL_JEPA(nn.Module):
    """
    VL-JEPA: Vision-Language Joint Embedding Predictive Architecture
    
    Ana mimari:
    1. X-Encoder: Xv → Sv (görsel embedding)
    2. Query Embedding: Xq → Sq (sorgu embedding)
    3. Predictor: (Sv, Sq) → Ŝy (tahmin edilen hedef)
    4. Y-Encoder: Y → Sy (gerçek hedef embedding)
    5. Loss: Alignment(Ŝy, Sy) + Regularization
    """
    
    def __init__(
        self,
        vision_input_dim=768,       # ViT/CLIP output dim
        vocab_size=32000,
        d_model=256,
        r=32,
        x_encoder_layers=4,
        y_encoder_layers=4,
        predictor_layers=5,
        window_offsets=[1, 2, 4, 8, 12, 16],
        max_visual_tokens=256,
        max_text_tokens=512,
        dropout=0.1,
        alignment_temp=0.07        # Contrastive temperature
    ):
        super().__init__()
        
        self.d_model = d_model
        self.alignment_temp = alignment_temp
        
        # X-Encoder (Vision)
        self.x_encoder = XEncoder(
            input_dim=vision_input_dim,
            d_model=d_model,
            r=r,
            n_layers=x_encoder_layers,
            window_offsets=window_offsets[:4],
            max_seq_len=max_visual_tokens,
            dropout=dropout
        )
        
        # Query tokenization & embedding
        self.query_embedding = nn.Embedding(vocab_size, d_model)
        self.query_encoder = GrassmannEncoder(
            d_model=d_model,
            r=r,
            n_layers=2,  # Lightweight
            window_offsets=window_offsets[:2],
            max_seq_len=max_text_tokens,
            dropout=dropout
        )
        
        # Predictor
        self.predictor = Predictor(
            d_model=d_model,
            r=r,
            n_layers=predictor_layers,
            window_offsets=window_offsets,
            max_seq_len=max_visual_tokens + max_text_tokens,
            dropout=dropout
        )
        
        # Y-Encoder (Target)
        self.y_encoder = YEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            r=r,
            n_layers=y_encoder_layers,
            window_offsets=window_offsets[:4],
            max_seq_len=max_text_tokens,
            dropout=dropout
        )
        
        # Projections for loss computation
        self.prediction_head = nn.Linear(d_model, d_model)
        self.target_head = nn.Linear(d_model, d_model)
        
    def encode_vision(self, visual_features):
        """Encode visual input."""
        return self.x_encoder(visual_features)
    
    def encode_query(self, query_ids):
        """Encode text query."""
        query_embeds = self.query_embedding(query_ids)
        Sq = self.query_encoder(query_embeds)
        return Sq
    
    def encode_target(self, target_ids):
        """Encode target text."""
        return self.y_encoder(target_ids)
    
    def predict(self, Sv, Sq):
        """Predict target embedding from visual and query."""
        return self.predictor(Sv, Sq)
    
    def forward(self, visual_features, query_ids, target_ids=None):
        """
        Forward pass for training.
        
        Args:
            visual_features: (batch, num_patches, vision_dim) - Vision backbone output
            query_ids: (batch, query_len) - Query token IDs
            target_ids: (batch, target_len) - Target token IDs (for training)
        
        Returns:
            dict containing:
                - Sy_hat: Predicted target embedding
                - Sy: Actual target embedding (if target_ids provided)
                - loss: Total loss (if target_ids provided)
        """
        # Encode visual input
        Sv, Sv_pooled = self.encode_vision(visual_features)
        
        # Encode query
        Sq = self.encode_query(query_ids)
        
        # Predict target embedding
        Sy_hat = self.predict(Sv, Sq)
        
        result = {
            'Sy_hat': Sy_hat,
            'Sv': Sv,
            'Sv_pooled': Sv_pooled,
            'Sq': Sq
        }
        
        # If target provided, compute loss
        if target_ids is not None:
            Sy, Sy_seq = self.encode_target(target_ids)
            
            # Project for loss computation
            Sy_hat_proj = self.prediction_head(Sy_hat)
            Sy_proj = self.target_head(Sy)
            
            # Normalize
            Sy_hat_norm = F.normalize(Sy_hat_proj, dim=-1)
            Sy_norm = F.normalize(Sy_proj, dim=-1)
            
            # Alignment loss (cosine similarity)
            alignment_loss = 1.0 - (Sy_hat_norm * Sy_norm).sum(dim=-1).mean()
            
            # Regularization loss (variance + covariance)
            reg_loss = self._compute_regularization(Sy_hat_norm, Sy_norm)
            
            total_loss = alignment_loss + 0.1 * reg_loss
            
            result.update({
                'Sy': Sy,
                'Sy_seq': Sy_seq,
                'loss': total_loss,
                'alignment_loss': alignment_loss,
                'reg_loss': reg_loss
            })
        
        return result
    
    def _compute_regularization(self, z1, z2):
        """
        VICReg-style regularization: variance + covariance.
        """
        batch_size, dim = z1.shape
        
        # Variance regularization
        std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
        std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
        var_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
        
        # Covariance regularization
        z1_centered = z1 - z1.mean(dim=0)
        z2_centered = z2 - z2.mean(dim=0)
        
        cov_z1 = (z1_centered.T @ z1_centered) / (batch_size - 1)
        cov_z2 = (z2_centered.T @ z2_centered) / (batch_size - 1)
        
        # Off-diagonal covariance
        off_diag_z1 = cov_z1.flatten()[:-1].view(dim - 1, dim + 1)[:, 1:].flatten()
        off_diag_z2 = cov_z2.flatten()[:-1].view(dim - 1, dim + 1)[:, 1:].flatten()
        
        cov_loss = (off_diag_z1.pow(2).sum() + off_diag_z2.pow(2).sum()) / dim
        
        return var_loss + cov_loss
    
    @torch.no_grad()
    def inference(self, visual_features, query_ids):
        """
        Inference mode - sadece tahmin döndürür.
        
        Args:
            visual_features: Vision input
            query_ids: Query tokens
        
        Returns:
            Sy_hat: Predicted target embedding
        """
        self.eval()
        
        Sv, _ = self.encode_vision(visual_features)
        Sq = self.encode_query(query_ids)
        Sy_hat = self.predict(Sv, Sq)
        
        return Sy_hat


# ============================================================================
# Legacy classes for backward compatibility
# ============================================================================

class BidirectionalGrassmannTransformer(nn.Module):
    """
    Bidirectional Grassmann Transformer dil modeli.
    Hem geçmiş hem gelecek bağlamı kullanarak daha zengin temsiller öğrenir.
    """
    
    def __init__(
        self,
        vocab_size=32000,
        d_model=256,
        r=32,
        n_layers=6,
        d_ff=1024,
        window_offsets=[1, 2, 4, 8, 12, 16],
        max_seq_len=512,
        dropout=0.1,
        **kwargs  # Accept additional kwargs for compatibility
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = GrassmannEncoder(
            d_model=d_model,
            r=r,
            n_layers=n_layers,
            window_offsets=window_offsets,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids):
        x = self.token_embedding(input_ids)
        hidden_states = self.encoder(x)
        logits = self.output_projection(hidden_states)
        return logits


# Backward compatibility aliases
CausalGrassmannTransformer = BidirectionalGrassmannTransformer