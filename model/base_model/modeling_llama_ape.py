"""
LLaMA model with Absolute Position Embeddings (APE) - No RoPE.

This is a modified LLaMA architecture that replaces Rotary Position Embeddings
with learned absolute position embeddings like GPT-2. This may help with
position-sensitive tasks where absolute token positions matter.

Key differences from standard LLaMA:
- Uses learned position embeddings added at input (like GPT)
- No RoPE application in attention
- Keeps RMSNorm, SiLU activation, and other LLaMA components
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LlamaAPEConfig:
    """Configuration for LLaMA with Absolute Position Embeddings."""
    vocab_size: int = 32000
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: Optional[int] = None
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = True
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    
    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (from LLaMA)."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class LlamaAPEAttention(nn.Module):
    """Multi-head attention WITHOUT rotary position embeddings."""
    
    def __init__(self, config: LlamaAPEConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        
        # Causal mask - will be created on first forward pass
        self.register_buffer("causal_mask", None, persistent=False)
    
    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get or create causal attention mask."""
        if self.causal_mask is None or self.causal_mask.size(0) < seq_len:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
            self.causal_mask = mask.masked_fill(mask == 1, float('-inf'))
        return self.causal_mask[:seq_len, :seq_len]
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Repeat K, V for grouped query attention
        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # Scaled dot-product attention with causal mask
        # Using Flash Attention if available (PyTorch 2.0+)
        if hasattr(F, 'scaled_dot_product_attention'):
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=None,
                dropout_p=self.config.attention_dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            # Manual attention computation
            scale = 1.0 / math.sqrt(self.head_dim)
            attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * scale
            
            # Apply causal mask
            causal_mask = self._get_causal_mask(seq_len, hidden_states.device)
            attn_weights = attn_weights + causal_mask
            
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.attention_dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        return self.o_proj(attn_output)


class LlamaAPEMLP(nn.Module):
    """LLaMA MLP with SwiGLU activation."""
    
    def __init__(self, config: LlamaAPEConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaAPEBlock(nn.Module):
    """LLaMA transformer block with APE attention."""
    
    def __init__(self, config: LlamaAPEConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = LlamaAPEAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = LlamaAPEMLP(config)
        self.hidden_dropout = nn.Dropout(config.hidden_dropout)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = self.hidden_dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.hidden_dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class LlamaAPEForCausalLM(nn.Module):
    """
    LLaMA model with Absolute Position Embeddings for causal language modeling.
    
    This model uses learned position embeddings (like GPT) instead of RoPE.
    """
    
    def __init__(self, config: LlamaAPEConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Learned position embeddings (like GPT)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Dropout after embeddings
        self.embed_dropout = nn.Dropout(config.hidden_dropout)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            LlamaAPEBlock(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Language model head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Weight tying
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Report parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"LlamaAPE model initialized with {n_params/1e6:.2f}M parameters")
    
    def _init_weights(self, module):
        """Initialize weights like LLaMA/GPT."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            input_ids: Token indices (batch_size, seq_len)
            labels: Target token indices for loss computation
        
        Returns:
            logits: Output logits (batch_size, seq_len, vocab_size)
            loss: Cross-entropy loss if labels provided
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_embeds = self.embed_tokens(input_ids)
        
        # Position embeddings
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embeddings(position_ids)
        
        # Combine embeddings
        hidden_states = token_embeds + position_embeds
        hidden_states = self.embed_dropout(hidden_states)
        
        # Transformer blocks
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        # Compute logits
        if labels is not None:
            # Training: compute all logits
            logits = self.lm_head(hidden_states)
            # Compute loss (ignore_index=-100 for padding)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        else:
            # Inference: only compute last position
            logits = self.lm_head(hidden_states[:, [-1], :])
            loss = None
        
        # Return in a format compatible with HuggingFace
        class Output:
            def __init__(self, logits, loss):
                self.logits = logits
                self.loss = loss
        
        return Output(logits, loss)
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively (matching GPT's generate).
        """
        for _ in range(max_new_tokens):
            # Crop if sequence is too long
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_position_embeddings else input_ids[:, -self.config.max_position_embeddings:]
            
            # Forward pass
            outputs = self(idx_cond)
            logits = outputs.logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat((input_ids, idx_next), dim=1)
        
        return input_ids


def create_llama_ape(
    vocab_size: int,
    hidden_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    max_position_embeddings: int,
    intermediate_size: Optional[int] = None,
    num_key_value_heads: Optional[int] = None,
    **kwargs
) -> LlamaAPEForCausalLM:
    """
    Helper function to create LlamaAPE model.
    """
    if intermediate_size is None:
        intermediate_size = hidden_size * 4
    
    config = LlamaAPEConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        **kwargs
    )
    
    return LlamaAPEForCausalLM(config)
