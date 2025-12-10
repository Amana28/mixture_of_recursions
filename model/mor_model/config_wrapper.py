"""
Configuration wrapper for MoR models.

This module provides a simple config class that mimics the Hydra DictConfig
interface used by the MoR models, allowing them to be used without Hydra.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


class DotDict(dict):
    """A dictionary that allows dot notation access."""
    
    def __getattr__(self, key):
        try:
            value = self[key]
            if isinstance(value, dict) and not isinstance(value, DotDict):
                value = DotDict(value)
                self[key] = value
            return value
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value
    
    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
    
    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def create_mor_config(
    mor_type: str = "expert",  # "expert" or "token"
    num_recursion: int = 2,
    capacity: str = "0.5,0.5",  # for expert choice
    sharing: str = "middle_cycle",
    router_type: str = "linear",
    router_temp: float = 1.0,
    # Expert-specific
    expert_alpha: float = 0.1,
    expert_router_func: str = "sigmoid",
    expert_sampling: str = "aux_loss",
    expert_aux_loss_coeff: float = 0.001,
    expert_cap_warmup_step: int = 500,
    expert_gating: str = "weighted",
    # Token-specific
    token_alpha: float = 1.0,
    token_router_func: str = "softmax",
    token_balancing: str = "loss",
    token_bal_loss_coeff: float = 0.1,
    token_bal_warmup_step: int = 0,
    token_gating: str = "weighted",
    # Common
    rand_router: bool = False,
    z_loss: float = 0.0,
    # Training params (for warmup calculations)
    num_warmup_steps: int = 500,
    gradient_accumulation_steps: int = 1,
    torch_dtype: str = "bfloat16",
    # KV sharing (optional)
    kv_sharing_enable: bool = False,
    kv_sharing_update_cache: bool = False,
) -> DotDict:
    """
    Create a MoR configuration dict that mimics Hydra's DictConfig.
    
    Args:
        mor_type: "expert" or "token"
        num_recursion: Number of recursion levels
        capacity: Comma-separated capacity factors for expert choice
        sharing: Layer sharing pattern ("cycle" or "middle_cycle")
        router_type: Type of router ("linear", "mlp", "wide_mlp")
        router_temp: Temperature for router
        expert_alpha: Alpha for expert choice
        expert_router_func: Router function for expert ("sigmoid", "tanh")
        expert_sampling: Sampling method for expert ("aux_loss", "aux_router")
        expert_aux_loss_coeff: Auxiliary loss coefficient for expert
        expert_cap_warmup_step: Capacity warmup steps for expert
        token_alpha: Alpha for token choice
        token_router_func: Router function for token ("softmax", "sigmoid")
        token_balancing: Balancing method for token ("loss", "loss_free")
        token_bal_loss_coeff: Balancing loss coefficient for token
        token_bal_warmup_step: Balancing warmup steps for token
        rand_router: Use random router (for ablation)
        z_loss: Z-loss coefficient
        num_warmup_steps: Number of warmup steps
        gradient_accumulation_steps: Gradient accumulation steps
        torch_dtype: Torch dtype string
        kv_sharing_enable: Enable KV sharing
        kv_sharing_update_cache: Update KV cache during forward
    
    Returns:
        DotDict mimicking Hydra config
    """
    cfg = DotDict({
        "mor": DotDict({
            "enable": True,
            "type": mor_type,
            "capacity": capacity,
            "router_type": router_type,
            "temp": router_temp,
            "rand_router": rand_router,
            "z_loss": z_loss,
            "expert": DotDict({
                "alpha": expert_alpha,
                "router_func": expert_router_func,
                "sampling": expert_sampling,
                "aux_loss_coeff": expert_aux_loss_coeff,
                "cap_warmup_step": expert_cap_warmup_step,
                "gating": expert_gating,
            }),
            "token": DotDict({
                "alpha": token_alpha,
                "router_func": token_router_func,
                "balancing": token_balancing,
                "bal_loss_coeff": token_bal_loss_coeff,
                "bal_warmup_step": token_bal_warmup_step,
                "gating": token_gating,
            }),
        }),
        "recursive": DotDict({
            "enable": True,
            "num_recursion": num_recursion,
            "sharing": sharing,
        }),
        "kv_sharing": DotDict({
            "enable": kv_sharing_enable,
            "num_recursion": num_recursion,
            "sharing": sharing,
            "update_cache": kv_sharing_update_cache,
        }),
        "num_warmup_steps": num_warmup_steps,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "torch_dtype": torch_dtype,
        "precision": "bf16" if torch_dtype == "bfloat16" else ("fp16" if torch_dtype == "float16" else "fp32"),
    })
    
    return cfg


def get_torch_dtype_from_config(cfg: DotDict):
    """Get torch dtype from config string."""
    import torch
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(cfg.get("torch_dtype", "bfloat16"), torch.bfloat16)
