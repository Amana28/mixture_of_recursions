"""
Training script for MoR (Mixture of Recursions) with Absolute Position Embeddings.

Supports both expert-choice and token-choice routing via --mor_type parameter.

Example usage:
    # Expert-choice MoR
    python train_scripts/train_mor_ape.py \
        --dataset dag/st \
        --mor_type expert \
        --num_recursion 2 \
        --n_layer 6 \
        --n_head 2 \
        --n_embd 240 \
        --max_iters 10000 \
        --num_nodes 100 \
        --num_of_paths 20

    # Token-choice MoR
    python train_scripts/train_mor_ape.py \
        --dataset dag/st \
        --mor_type token \
        --num_recursion 2 \
        --n_layer 6 \
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

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.mor_model.modeling_mor_ape import MoRAPEConfig, MoRAPEForCausalLM, create_mor_ape

# -----------------------------------------------------------------------------
# Logging setup

def get_logger(filename, verbosity=0, name=None):
    """Create a logger that writes to file only."""
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    logger.propagate = False
    logger.handlers = []
    
    fh = logging.FileHandler(filename, "w")
    fh.setLevel(level_dict[verbosity])
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger


# -----------------------------------------------------------------------------
# Parse arguments

parser = argparse.ArgumentParser(description='Training MoR-APE on graph data.')

parser.add_argument('--dataset', type=str, default='dag/st', help='Dataset path')
parser.add_argument('--mor_type', type=str, default='expert', choices=['expert', 'token'],
                    help='MoR routing type: expert or token')
parser.add_argument('--num_recursion', type=int, default=2, help='Number of recursions')
parser.add_argument('--capacity', type=str, default='0.5,0.5', 
                    help='Capacity factors for expert-choice (comma-separated)')
parser.add_argument('--sharing', type=str, default='middle_cycle', choices=['middle_cycle', 'cycle'],
                    help='Sharing strategy: middle_cycle (first+last layers no routing) or cycle (all routing)')
parser.add_argument('--aux_loss_coeff', type=float, default=0.001,
                    help='Auxiliary router BCE loss coefficient (0 to disable)')
parser.add_argument('--cap_warmup_steps', type=int, default=0,
                    help='Capacity warmup steps (0 to disable, cosine decay from 1.0 to target)')
parser.add_argument('--save_interval', type=int, default=1000,
                    help='Save checkpoint every N iterations')
parser.add_argument('--n_layer', type=int, default=6, help='Number of layers (should be divisible)')
parser.add_argument('--n_head', type=int, default=2, help='Number of attention heads')
parser.add_argument('--n_embd', type=int, default=240, help='Embedding size')
parser.add_argument('--max_iters', type=int, default=10000, help='Number of iterations')
parser.add_argument('--num_nodes', type=int, default=100, help='Number of nodes')
parser.add_argument('--num_of_paths', type=int, default=20, help='Number of paths')
parser.add_argument('--train_batch_size', type=int, default=64, help='Training batch size')
parser.add_argument('--val_batch_size', type=int, default=64, help='Validation batch size')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate')
parser.add_argument('--compile', action='store_true', help='Use torch.compile')
parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu/mps)')

args = parser.parse_args()

# -----------------------------------------------------------------------------
# Configuration

dataset = args.dataset
mor_type = args.mor_type
num_recursion = args.num_recursion
capacity_factors = [float(c) for c in args.capacity.split(',')]
sharing_strategy = args.sharing
n_layer = args.n_layer
n_head = args.n_head
n_embd = args.n_embd
max_iters = args.max_iters
num_nodes = args.num_nodes
num_of_paths = args.num_of_paths
train_batch_size = args.train_batch_size
val_batch_size = args.val_batch_size
learning_rate = args.learning_rate

# Validate layer count based on sharing strategy
if sharing_strategy == "middle_cycle":
    # middle_cycle: n_layer = 2 + (base_depth * num_recursion)
    base_depth = (n_layer - 2) // num_recursion
    if (n_layer - 2) % num_recursion != 0:
        print(f"Warning: n_layer-2 ({n_layer-2}) is not divisible by num_recursion ({num_recursion})")
        print(f"Effective layers per recursion: {base_depth}")
else:  # cycle
    # cycle: n_layer = base_depth * num_recursion
    base_depth = n_layer // num_recursion
    if n_layer % num_recursion != 0:
        print(f"Warning: n_layer ({n_layer}) is not divisible by num_recursion ({num_recursion})")
        print(f"Effective layers per recursion: {base_depth}")

# Validate capacity factors for expert-choice
if mor_type == "expert" and len(capacity_factors) != num_recursion:
    print(f"⚠️  WARNING: capacity has {len(capacity_factors)} values but num_recursion is {num_recursion}")
    print(f"⚠️  Missing recursions will use default capacity=0.5")
    print(f"⚠️  Recommended: --capacity {','.join(['0.25'] * num_recursion)} for {num_recursion} recursions")

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
out_dir = f'out/{dataset}_mor_{mor_type}_ape_{n_layer}L_{n_head}H_{n_embd}E_{num_recursion}R_{num_nodes}N'
os.makedirs(out_dir, exist_ok=True)

# Initialize logger
if num_of_paths == 0:
    log_file_name = os.path.join(out_dir, 'train.log')
else:
    log_file_name = os.path.join(out_dir, f'train_{num_of_paths}.log')
logger = get_logger(log_file_name, verbosity=1, name='train_mor_ape')

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
    
    # Replace PAD tokens (0) with -100 for loss computation
    y = torch.where(y == 0, torch.tensor(-100), y)
    
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# -----------------------------------------------------------------------------
# Model initialization

print(f"Initializing MoR-APE model ({mor_type}-choice, {num_recursion} recursions, {sharing_strategy})...")

model = create_mor_ape(
    vocab_size=vocab_size,
    hidden_size=n_embd,
    intermediate_size=n_embd * 4,
    num_hidden_layers=n_layer,
    num_attention_heads=n_head,
    max_position_embeddings=block_size,
    mor_type=mor_type,
    num_recursion=num_recursion,
    capacity_factors=capacity_factors,
    sharing_strategy=sharing_strategy,
    expert_aux_loss_coeff=args.aux_loss_coeff,
    expert_cap_warmup_steps=args.cap_warmup_steps,
    attention_dropout=dropout,
    hidden_dropout=dropout,
)

model.to(device)

# Compile if requested
if args.compile and hasattr(torch, 'compile'):
    print("Compiling model with torch.compile...")
    model = torch.compile(model)

# -----------------------------------------------------------------------------
# Optimizer

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
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16' and device_type == 'cuda'))

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
        'config': {
            'vocab_size': vocab_size,
            'hidden_size': n_embd,
            'intermediate_size': n_embd * 4,
            'num_hidden_layers': n_layer,
            'num_attention_heads': n_head,
            'max_position_embeddings': block_size,
            'mor_type': mor_type,
            'num_recursion': num_recursion,
            'capacity_factors': capacity_factors,
            'sharing_strategy': sharing_strategy,
        },
        'args': vars(args),
    }
    ckpt_name = f'{iter_num}_ckpt.pt' if num_of_paths == 0 else f'{iter_num}_ckpt_{num_of_paths}.pt'
    torch.save(checkpoint, os.path.join(out_dir, ckpt_name))
    print(f"Saved checkpoint to {out_dir}/{ckpt_name}")


# -----------------------------------------------------------------------------
# Training loop

print(f"\nStarting training for {max_iters} iterations...")
print(f"MoR type: {mor_type}, Recursions: {num_recursion}, Capacity: {capacity_factors}")
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
    
    # Save checkpoint at intervals
    if iter_num > 0 and iter_num % args.save_interval == 0:
        save_checkpoint(iter_num, best_val_loss)
        print(f"Saved checkpoint at iter {iter_num}")
        logger.info(f"Saved checkpoint at iter {iter_num}")

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
