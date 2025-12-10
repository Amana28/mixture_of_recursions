"""
Training script for Vanilla LLaMA model using custom bin file data.

Example usage:
    python train_scripts/train_vanilla.py \
        --dataset dag/st \
        --n_layer 2 \
        --n_head 2 \
        --n_embd 240 \
        --max_iters 10000 \
        --num_nodes 100 \
        --num_of_paths 20
"""

import os
import sys
import time
import math
import pickle
import logging
from contextlib import nullcontext
import argparse

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Use standard transformers LlamaForCausalLM (compatible with all transformers versions)
from transformers import LlamaConfig, LlamaForCausalLM

# -----------------------------------------------------------------------------
# Logging setup

def get_logger(filename, verbosity=0, name=None):
    """Create a logger that writes to file only (no console output)."""
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    logger.propagate = False  # Don't propagate to root logger (prevents console output)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # File handler only (no console handler)
    fh = logging.FileHandler(filename, "w")
    fh.setLevel(level_dict[verbosity])
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger


def open_and_append(filename, text):
    """Append text to a file."""
    with open(filename, 'a') as file:
        file.write(text + '\n')

# -----------------------------------------------------------------------------
# Parse arguments

parser = argparse.ArgumentParser(description='Training of LLaMA model on graph data.')

parser.add_argument('--dataset', type=str, default='simple_graph', help='Name of the dataset to use')  
parser.add_argument('--n_layer', type=int, default=2, help='Number of layers (default: 2)')  
parser.add_argument('--n_head', type=int, default=2, help='Number of attention heads (default: 2)')  
parser.add_argument('--n_embd', type=int, default=240, help='Size of the embeddings (default: 240)')
parser.add_argument('--max_iters', type=int, default=10000, help='Number of Iterations (default: 10000)')
parser.add_argument('--num_nodes', type=int, default=100, help='Number of Nodes (default: 100)')
parser.add_argument('--num_of_paths', type=int, default=20, help='Number of Paths (default: 20)')
parser.add_argument('--train_batch_size', type=int, default=64, help='Training batch size (default: 64)')
parser.add_argument('--val_batch_size', type=int, default=64, help='Validation batch size (default: 64)')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate (default: 5e-4)')
parser.add_argument('--compile', action='store_true', help='Use torch.compile (PyTorch 2.0+)')
parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu/mps)')

args = parser.parse_args()

# -----------------------------------------------------------------------------
# Configuration

dataset = args.dataset
n_layer = args.n_layer
n_head = args.n_head
n_embd = args.n_embd
max_iters = args.max_iters
num_nodes = args.num_nodes
num_of_paths = args.num_of_paths
train_batch_size = args.train_batch_size
val_batch_size = args.val_batch_size
learning_rate = args.learning_rate

# Data directory
data_dir = os.path.join('data', f'{dataset}/{num_nodes}')

# Load metadata
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
    
stoi, itos = meta['stoi'], meta['itos']
block_size = meta['block_size']
vocab_size = meta['vocab_size']

print(f"Vocabulary size: {vocab_size}")
print(f"Block size: {block_size}")

# Output directory
out_dir = f'out/{dataset}_llama_{n_layer}L_{n_head}H_{n_embd}E_{num_nodes}N'
os.makedirs(out_dir, exist_ok=True)

# Initialize logger
if num_of_paths == 0:
    log_file_name = os.path.join(out_dir, 'train.log')
else:
    log_file_name = os.path.join(out_dir, f'train_{num_of_paths}.log')
logger = get_logger(log_file_name, verbosity=1, name='train_vanilla')

# -----------------------------------------------------------------------------
# Training hyperparameters

eval_interval = max(1, max_iters // 10)
log_interval = max(1, max_iters // 100)
eval_iters = min(100, max_iters // 10)

gradient_accumulation_steps = 1
dropout = 0.0
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = max_iters // 20
lr_decay_iters = max_iters
min_lr = learning_rate / 10

# Device setup
device = args.device
if device == 'cuda' and not torch.cuda.is_available():
    print("CUDA not available, falling back to CPU")
    device = 'cpu'
elif device == 'mps' and not torch.backends.mps.is_available():
    print("MPS not available, falling back to CPU")
    device = 'cpu'

device_type = 'cuda' if 'cuda' in device else ('mps' if 'mps' in device else 'cpu')
dtype = 'bfloat16' if device_type == 'cuda' and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

print(f"Using device: {device}, dtype: {dtype}")

# -----------------------------------------------------------------------------
# Data loading

if num_of_paths == 0:
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
else:
    train_data = np.memmap(os.path.join(data_dir, f'train_{num_of_paths}.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

print(f"Train data size: {len(train_data):,} tokens")
print(f"Val data size: {len(val_data):,} tokens")


def get_batch(split):
    """Get a batch of data for training or validation."""
    data = train_data if split == 'train' else val_data
    batch_sz = train_batch_size if split == 'train' else val_batch_size
    
    data_size = block_size + 1
    ix = torch.randint((len(data) - data_size) // data_size, (batch_sz,)) * data_size
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    # CRITICAL: Replace PAD tokens (0) with -100 in labels
    # HuggingFace uses ignore_index=-100, original GPT uses ignore_index=0
    y = torch.where(y == 0, torch.tensor(-100), y)
    
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# -----------------------------------------------------------------------------
# Model initialization

print("Initializing LLaMA model...")

# Create LLaMA config
# For small models, we use the same number for num_attention_heads and num_key_value_heads (MHA)
# For larger models, you can reduce num_key_value_heads for GQA
config = LlamaConfig(
    vocab_size=vocab_size,
    hidden_size=n_embd,
    intermediate_size=n_embd * 4,  # Standard 4x multiplier for FFN
    num_hidden_layers=n_layer,
    num_attention_heads=n_head,
    num_key_value_heads=n_head,  # Same as n_head for MHA (can reduce for GQA)
    max_position_embeddings=block_size,
    rms_norm_eps=1e-6,
    pad_token_id=0,  # Your [PAD] token
    bos_token_id=None,
    eos_token_id=1,  # Your \n token as EOS
    tie_word_embeddings=True,
    use_cache=False,  # Disable KV cache for training
    attention_dropout=dropout,
    hidden_act="silu",
    initializer_range=0.02,
    attention_bias=False,
    mlp_bias=False,
)

model = LlamaForCausalLM(config)
model.to(device)

# Count parameters
n_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {n_params/1e6:.2f}M")

# Compile if requested (PyTorch 2.0+)
if args.compile and hasattr(torch, 'compile'):
    print("Compiling model with torch.compile...")
    model = torch.compile(model)

# -----------------------------------------------------------------------------
# Optimizer

# Separate parameters for weight decay
decay_params = []
no_decay_params = []
for name, param in model.named_parameters():
    if param.requires_grad:
        if 'bias' in name or 'norm' in name or 'embedding' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

optim_groups = [
    {"params": decay_params, "weight_decay": weight_decay},
    {"params": no_decay_params, "weight_decay": 0.0},
]

optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(beta1, beta2))

# Gradient scaler for mixed precision
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16' and device_type == 'cuda'))

# -----------------------------------------------------------------------------
# Learning rate scheduler

def get_lr(it):
    """Cosine learning rate schedule with warmup."""
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# -----------------------------------------------------------------------------
# Training utilities

@torch.no_grad()
def estimate_loss():
    """Estimate loss on train and val sets."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                outputs = model(input_ids=X, labels=Y)
                loss = outputs.loss
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def save_checkpoint(iter_num, best_val_loss):
    """Save model checkpoint."""
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': config.to_dict(),
        'args': vars(args),
    }
    ckpt_name = f'{iter_num}_ckpt.pt' if num_of_paths == 0 else f'{iter_num}_ckpt_{num_of_paths}.pt'
    torch.save(checkpoint, os.path.join(out_dir, ckpt_name))
    print(f"Saved checkpoint to {out_dir}/{ckpt_name}")


# -----------------------------------------------------------------------------
# Training loop

print(f"\nStarting training for {max_iters} iterations...")
print(f"Output directory: {out_dir}")
print("-" * 50)

iter_num = 0
best_val_loss = float('inf')
t0 = time.time()

X, Y = get_batch('train')

while iter_num <= max_iters:
    # Set learning rate
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate and save checkpoint
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        eval_msg = f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.2e}"
        print(eval_msg)
        logger.info(eval_msg)
        
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            save_checkpoint(iter_num, best_val_loss)
            print(f"Saved best checkpoint at iter {iter_num}")
            logger.info(f"Saved best checkpoint at iter {iter_num}")

    # Forward pass
    with ctx:
        outputs = model(input_ids=X, labels=Y)
        loss = outputs.loss / gradient_accumulation_steps

    # Get next batch
    X, Y = get_batch('train')

    # Backward pass
    scaler.scale(loss).backward()
    
    # Gradient clipping
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    
    if iter_num % log_interval == 0:
        lossf = loss.item() * gradient_accumulation_steps
        iter_msg = f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.2e}"
        print(iter_msg)
        logger.info(iter_msg)

    iter_num += 1

# Final save
save_checkpoint(iter_num - 1, best_val_loss)
logger.info(f"Training complete! Best val loss: {best_val_loss:.4f}")
logger.info(f"Checkpoints saved to: {out_dir}")
print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
print(f"Checkpoints saved to: {out_dir}")
