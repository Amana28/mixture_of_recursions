"""
MoR (Mixture of Recursions) model with Absolute Position Embeddings.

This module combines:
- MoR routing (expert-choice or token-choice)
- Absolute Position Embeddings (like GPT, no RoPE)
- LLaMA-style components (RMSNorm, SwiGLU)

Designed for standalone training on custom data without Hydra.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base_model.modeling_llama_ape import (
    LlamaAPEConfig,
    RMSNorm,
    LlamaAPEAttention,
    LlamaAPEMLP,
    LlamaAPEBlock,
    LlamaAPEForCausalLM,
)
from model.mor_model.util import ROUTER_TYPES, MoRLayerOutputWithPast


@dataclass
class MoRAPEConfig(LlamaAPEConfig):
    """Configuration for MoR with Absolute Position Embeddings."""
    # MoR settings
    mor_type: str = "expert"  # "expert" or "token"
    num_recursion: int = 2
    capacity_factors: List[float] = field(default_factory=lambda: [0.5, 0.5])
    sharing_strategy: str = "middle_cycle"  # "middle_cycle" or "cycle"
    router_type: str = "linear"
    router_temp: float = 1.0
    # Expert-choice settings
    expert_alpha: float = 0.1
    expert_router_func: str = "sigmoid"
    expert_gating: str = "weighted"
    # Token-choice settings
    token_alpha: float = 1.0
    token_router_func: str = "softmax"
    token_gating: str = "weighted"
    token_balancing: str = "loss"
    token_bal_loss_coeff: float = 0.1


class LinearRouter(nn.Module):
    """Simple linear router for MoR."""
    
    def __init__(self, hidden_size: int, out_dim: int = 1, initializer_range: float = 0.02):
        super().__init__()
        self.router = nn.Linear(hidden_size, out_dim, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=initializer_range)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.router(x)


class ExpertChoiceMoRBlock(nn.Module):
    """
    MoR block using expert-choice routing.
    
    The expert (block) selects which tokens to process based on router scores.
    """
    
    def __init__(
        self,
        config: MoRAPEConfig,
        blocks: nn.ModuleList,
        capacity_factor: float = 0.5,
    ):
        super().__init__()
        self.config = config
        self.blocks = blocks
        self.capacity_factor = capacity_factor
        
        # Router
        self.router = LinearRouter(
            config.hidden_size, 
            out_dim=1,
            initializer_range=config.initializer_range
        )
        
        self.router_func = config.expert_router_func
        self.alpha = config.expert_alpha
        self.gating = config.expert_gating
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        prev_selected_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size) - FULL sequence
            prev_selected_tokens: (batch_size, prev_k, 1) - indices from previous block
        
        Returns:
            output: Updated hidden states (full sequence)
            router_loss: Router z-loss
            selected_tokens: Current selection indices for chaining
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # If we have previous selection, gather those tokens first
        if prev_selected_tokens is not None:
            x = torch.gather(hidden_states, 1, 
                            index=prev_selected_tokens.expand(-1, -1, hidden_dim))
            current_seq_len = x.shape[1]
        else:
            x = hidden_states
            current_seq_len = seq_len
        
        # Compute router logits on current token set
        router_logits = self.router(x / self.config.router_temp)
        
        # Apply router function
        if self.router_func == "sigmoid":
            router_weights = torch.sigmoid(router_logits)
            router_probs = router_weights * self.alpha
        elif self.router_func == "tanh":
            router_weights = torch.tanh(router_logits)
            router_probs = router_weights * self.alpha
        else:
            router_probs = router_weights = router_logits
        
        # Select top-k tokens from current set
        top_k = max(1, int(self.capacity_factor * current_seq_len))
        weights, selected_indices = torch.topk(router_probs, top_k, dim=1, sorted=False)
        
        # Sort indices to maintain causal order
        selected_indices, sort_idx = torch.sort(selected_indices, dim=1)
        weights = torch.gather(weights, dim=1, index=sort_idx)
        
        # Gather selected tokens from x (the current working set)
        indices_expanded = selected_indices.expand(-1, -1, hidden_dim)
        selected_tokens = torch.gather(x, dim=1, index=indices_expanded)
        
        # Process through blocks
        for block in self.blocks:
            selected_tokens = block(selected_tokens)
        
        # Apply gating
        if self.gating == "weighted":
            processed = selected_tokens * weights
        else:
            processed = selected_tokens
        
        # Map selected_indices back to original sequence positions
        if prev_selected_tokens is not None:
            # selected_indices are relative to prev_selected_tokens
            # Map back to original positions
            orig_indices = torch.gather(prev_selected_tokens, dim=1, index=selected_indices)
        else:
            orig_indices = selected_indices
        
        # Scatter back to original positions in full sequence
        orig_indices_expanded = orig_indices.expand(-1, -1, hidden_dim)
        output = hidden_states.clone()
        output = torch.scatter_add(
            output, dim=1, index=orig_indices_expanded, src=processed
        )
        
        # Router z-loss for regularization
        router_z_loss = torch.logsumexp(router_logits.squeeze(-1), dim=-1)
        router_z_loss = router_z_loss.pow(2).mean()
        
        return output, router_z_loss, orig_indices


class TokenChoiceMoRBlock(nn.Module):
    """
    MoR block using token-choice routing.
    
    Each token chooses which recursion level to use.
    """
    
    def __init__(
        self,
        config: MoRAPEConfig,
        block_lists: nn.ModuleList,  # List of block lists, one per recursion
    ):
        super().__init__()
        self.config = config
        self.block_lists = block_lists
        self.num_recursion = config.num_recursion
        
        # Router outputs scores for each recursion level
        self.router = LinearRouter(
            config.hidden_size,
            out_dim=config.num_recursion,
            initializer_range=config.initializer_range
        )
        
        self.router_func = config.token_router_func
        self.alpha = config.token_alpha
        self.gating = config.token_gating
        self.balancing = config.token_balancing
        self.bal_loss_coeff = config.token_bal_loss_coeff
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
        
        Returns:
            output: Processed hidden states
            balancing_loss: Optional load balancing loss
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Compute router logits
        router_logits = self.router(hidden_states / self.config.router_temp)
        
        # Apply router function to get probabilities
        if self.router_func == "softmax":
            router_probs = F.softmax(router_logits, dim=-1) * self.alpha
        elif self.router_func == "sigmoid":
            router_probs = torch.sigmoid(router_logits) * self.alpha
        else:
            router_probs = router_logits
        
        # Each token picks top-1 recursion level
        weights, expert_indices = torch.topk(router_probs, 1, dim=-1, sorted=False)
        expert_indices = expert_indices.squeeze(-1)  # (batch_size, seq_len)
        
        # Initialize output
        output = hidden_states.clone()
        
        # Process each recursion level
        for level, blocks in enumerate(self.block_lists):
            # Find tokens assigned to this level
            mask = (expert_indices == level)
            
            if not mask.any():
                continue
            
            # Extract tokens for this level
            level_tokens = []
            level_indices = []
            for b in range(batch_size):
                token_mask = mask[b]
                if token_mask.any():
                    level_tokens.append(hidden_states[b, token_mask])
                    level_indices.append((b, token_mask.nonzero().squeeze(-1)))
            
            if len(level_tokens) == 0:
                continue
            
            # Pad and batch tokens
            max_len = max(t.size(0) for t in level_tokens)
            batched = torch.zeros(len(level_tokens), max_len, hidden_dim, 
                                  device=hidden_states.device, dtype=hidden_states.dtype)
            for i, t in enumerate(level_tokens):
                batched[i, :t.size(0)] = t
            
            # Process through blocks at this level
            for block in blocks:
                batched = block(batched)
            
            # Scatter back
            for i, (b, indices) in enumerate(level_indices):
                # Get gating weights for these tokens
                token_weights = weights[b, indices]
                processed = batched[i, :len(indices)]
                
                if self.gating == "weighted":
                    processed = processed * token_weights
                
                output[b, indices] = output[b, indices] + processed
        
        # Compute balancing loss if training
        balancing_loss = None
        if self.training and self.balancing == "loss":
            # Load balance across experts
            expert_probs = router_probs.mean(dim=(0, 1))  # Average prob per expert
            expert_counts = torch.bincount(expert_indices.view(-1), 
                                          minlength=self.num_recursion).float()
            expert_freq = expert_counts / (batch_size * seq_len)
            balancing_loss = (expert_probs * expert_freq).sum() * self.bal_loss_coeff
        
        # Router z-loss
        router_z_loss = torch.logsumexp(router_logits, dim=-1)
        router_z_loss = router_z_loss.pow(2).mean()
        
        return output, router_z_loss, balancing_loss


class MoRAPEForCausalLM(nn.Module):
    """
    MoR (Mixture of Recursions) model with Absolute Position Embeddings.
    
    Supports both expert-choice and token-choice routing.
    """
    
    def __init__(self, config: MoRAPEConfig):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.embed_dropout = nn.Dropout(config.hidden_dropout)
        
        # Build layers based on MoR type
        self._build_layers(config)
        
        # Final norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # LM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Weight tying
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        
        # Initialize
        self.apply(self._init_weights)
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"MoR-APE ({config.mor_type}) initialized with {n_params/1e6:.2f}M parameters")
    
    def _build_layers(self, config: MoRAPEConfig):
        """Build transformer layers with MoR routing."""
        layers = []
        
        if config.sharing_strategy == "middle_cycle":
            # middle_cycle: [first_layer, MoR blocks, last_layer]
            base_depth = (config.num_hidden_layers - 2) // config.num_recursion
            
            # First layer (no MoR)
            layers.append(LlamaAPEBlock(config))
            
            if config.mor_type == "expert":
                for rec_idx in range(config.num_recursion):
                    blocks = nn.ModuleList([
                        LlamaAPEBlock(config) for _ in range(base_depth)
                    ])
                    capacity = config.capacity_factors[rec_idx] if rec_idx < len(config.capacity_factors) else 0.5
                    layers.append(ExpertChoiceMoRBlock(config, blocks, capacity))
            
            elif config.mor_type == "token":
                block_lists = nn.ModuleList([
                    nn.ModuleList([LlamaAPEBlock(config) for _ in range(base_depth)])
                    for _ in range(config.num_recursion)
                ])
                layers.append(TokenChoiceMoRBlock(config, block_lists))
            
            # Last layer (no MoR)
            layers.append(LlamaAPEBlock(config))
        
        elif config.sharing_strategy == "cycle":
            # cycle: [MoR blocks only] - all layers use routing
            base_depth = config.num_hidden_layers // config.num_recursion
            
            if config.mor_type == "expert":
                for rec_idx in range(config.num_recursion):
                    blocks = nn.ModuleList([
                        LlamaAPEBlock(config) for _ in range(base_depth)
                    ])
                    capacity = config.capacity_factors[rec_idx] if rec_idx < len(config.capacity_factors) else 0.5
                    layers.append(ExpertChoiceMoRBlock(config, blocks, capacity))
            
            elif config.mor_type == "token":
                block_lists = nn.ModuleList([
                    nn.ModuleList([LlamaAPEBlock(config) for _ in range(base_depth)])
                    for _ in range(config.num_recursion)
                ])
                layers.append(TokenChoiceMoRBlock(config, block_lists))
        
        else:
            raise ValueError(f"Unknown sharing strategy: {config.sharing_strategy}")
        
        self.layers = nn.ModuleList(layers)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass.
        
        Returns an object with .loss and .logits attributes.
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_embeds = self.embed_tokens(input_ids)
        
        # Position embeddings
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embeddings(position_ids)
        
        # Combine
        hidden_states = token_embeds + position_embeds
        hidden_states = self.embed_dropout(hidden_states)
        
        # Track auxiliary losses
        total_router_z_loss = 0.0
        total_balancing_loss = 0.0
        
        # Track selected tokens for chaining expert-choice
        prev_selected_tokens = None
        
        # Process through layers
        for layer in self.layers:
            if isinstance(layer, (ExpertChoiceMoRBlock, TokenChoiceMoRBlock)):
                if isinstance(layer, ExpertChoiceMoRBlock):
                    hidden_states, router_z_loss, selected_tokens = layer(
                        hidden_states, prev_selected_tokens
                    )
                    total_router_z_loss = total_router_z_loss + router_z_loss
                    prev_selected_tokens = selected_tokens  # Chain to next block
                else:
                    hidden_states, router_z_loss, bal_loss = layer(hidden_states)
                    total_router_z_loss = total_router_z_loss + router_z_loss
                    if bal_loss is not None:
                        total_balancing_loss = total_balancing_loss + bal_loss
            else:
                hidden_states = layer(hidden_states)
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        # Compute logits and loss
        if labels is not None:
            logits = self.lm_head(hidden_states)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            # Add auxiliary losses
            if self.training:
                loss = loss + 1e-5 * total_router_z_loss
                if total_balancing_loss != 0.0:
                    loss = loss + total_balancing_loss
        else:
            logits = self.lm_head(hidden_states[:, [-1], :])
            loss = None
        
        # Return in HuggingFace-compatible format
        class Output:
            def __init__(self, logits, loss, router_z_loss=None, balancing_loss=None):
                self.logits = logits
                self.loss = loss
                self.router_z_loss = router_z_loss
                self.balancing_loss = balancing_loss
        
        return Output(logits, loss, total_router_z_loss, total_balancing_loss)
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate tokens autoregressively."""
        for _ in range(max_new_tokens):
            # Crop if too long
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_position_embeddings else input_ids[:, -self.config.max_position_embeddings:]
            
            # Forward
            outputs = self(idx_cond)
            logits = outputs.logits[:, -1, :] / temperature
            
            # Top-k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat((input_ids, idx_next), dim=1)
        
        return input_ids


def create_mor_ape(
    vocab_size: int,
    hidden_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    max_position_embeddings: int,
    mor_type: str = "expert",
    num_recursion: int = 2,
    capacity_factors: List[float] = None,
    intermediate_size: Optional[int] = None,
    **kwargs
) -> MoRAPEForCausalLM:
    """
    Helper to create MoR-APE model.
    
    Args:
        mor_type: "expert" or "token"
        num_recursion: Number of recursion levels
        capacity_factors: List of capacity factors for each recursion (expert-choice)
    """
    if intermediate_size is None:
        intermediate_size = hidden_size * 4
    
    if capacity_factors is None:
        capacity_factors = [1.0 / num_recursion] * num_recursion
    
    config = MoRAPEConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        max_position_embeddings=max_position_embeddings,
        mor_type=mor_type,
        num_recursion=num_recursion,
        capacity_factors=capacity_factors,
        **kwargs
    )
    
    return MoRAPEForCausalLM(config)
