"""
LLaMA model with learned positional embeddings (like GPT).

This wraps LlamaForCausalLM and adds learned position embeddings
at the input layer, similar to GPT's architecture.
"""

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM


class LlamaWithLearnedPE(nn.Module):
    """
    LLaMA model with learned positional embeddings added at input.
    
    This combines:
    - Learned absolute position embeddings (like GPT)
    - LLaMA's architecture (RMSNorm, SiLU, etc.)
    
    Note: RoPE still runs inside attention, but the learned PEs
    provide additional absolute position signal.
    """
    
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        
        # Create standard LLaMA model
        self.llama = LlamaForCausalLM(config)
        
        # Add learned positional embeddings (like GPT's wpe)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, 
            config.hidden_size
        )
        
        # Initialize position embeddings
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
        
    def forward(self, input_ids, labels=None, **kwargs):
        """
        Forward pass with learned position embeddings added.
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        
        # Get token embeddings from LLaMA's embed_tokens
        token_embeddings = self.llama.model.embed_tokens(input_ids)
        
        # Create position indices and get position embeddings
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)
        
        # Add position embeddings to token embeddings
        inputs_embeds = token_embeddings + position_embeddings
        
        # Forward through LLaMA using inputs_embeds instead of input_ids
        outputs = self.llama(
            inputs_embeds=inputs_embeds,
            labels=labels,
            **kwargs
        )
        
        return outputs
    
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate tokens autoregressively (matching GPT's generate).
        """
        device = input_ids.device
        
        for _ in range(max_new_tokens):
            # Get current sequence length
            seq_len = input_ids.size(1)
            
            # Get embeddings
            token_embeddings = self.llama.model.embed_tokens(input_ids)
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_ids.size(0), -1)
            position_embeddings = self.position_embeddings(position_ids)
            inputs_embeds = token_embeddings + position_embeddings
            
            # Forward pass
            outputs = self.llama(inputs_embeds=inputs_embeds)
            logits = outputs.logits
            
            # Get logits for last position and scale by temperature
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample from distribution
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat((input_ids, idx_next), dim=1)
        
        return input_ids
    
    def state_dict(self, *args, **kwargs):
        """Return combined state dict."""
        state = {}
        # Add LLaMA state
        for k, v in self.llama.state_dict().items():
            state[f'llama.{k}'] = v
        # Add position embeddings
        state['position_embeddings.weight'] = self.position_embeddings.weight
        return state
    
    def load_state_dict(self, state_dict, strict=True):
        """Load state dict."""
        # Separate LLaMA and position embedding states
        llama_state = {}
        for k, v in state_dict.items():
            if k.startswith('llama.'):
                llama_state[k[6:]] = v  # Remove 'llama.' prefix
            elif k == 'position_embeddings.weight':
                self.position_embeddings.weight.data.copy_(v)
        
        self.llama.load_state_dict(llama_state, strict=strict)
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def to(self, device):
        super().to(device)
        return self


def create_llama_with_learned_pe(
    vocab_size,
    hidden_size,
    num_hidden_layers,
    num_attention_heads,
    max_position_embeddings,
    intermediate_size=None,
    num_key_value_heads=None,
    **kwargs
):
    """
    Helper function to create LlamaWithLearnedPE model.
    """
    if intermediate_size is None:
        intermediate_size = hidden_size * 4
    if num_key_value_heads is None:
        num_key_value_heads = num_attention_heads
    
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=1e-6,
        pad_token_id=0,
        tie_word_embeddings=True,
        use_cache=False,
        attention_dropout=0.0,
        hidden_act="silu",
        initializer_range=0.02,
        attention_bias=False,
        mlp_bias=False,
        **kwargs
    )
    
    return LlamaWithLearnedPE(config)
