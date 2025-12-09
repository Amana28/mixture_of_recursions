"""
Training script for Graph Path Prediction (Refactored Style)
"""
import os
# Suppress TF/XLA warnings immediately
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import time
import math
import pickle
from contextlib import nullcontext
import argparse
import numpy as np
import torch
import logging
from transformers import LlamaConfig, LlamaForCausalLM
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Input Parameters
parser = argparse.ArgumentParser(description='Training of Graph Path Predictor')
parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
parser.add_argument('--output_dir', type=str, default='checkpoints/graph_bin_model', help='Output directory')
parser.add_argument('--num_train_epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='Max learning rate')
parser.add_argument('--per_device_train_batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--block_size', type=int, default=64, help='Context length')
parser.add_argument('--save_steps', type=int, default=1000, help='Save checkpoint every N steps')
parser.add_argument('--eval_steps', type=int, default=1000, help='Evaluate every N steps')
parser.add_argument('--log_interval', type=int, default=100, help='Log every N steps')
# Model size arguments
parser.add_argument('--hidden_size', type=int, default=256, help='Model hidden size')
parser.add_argument('--num_layers', type=int, default=8, help='Number of transformer layers')
parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
# Regularization arguments
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
args = parser.parse_args()

# Configuration
data_dir = args.data_dir
out_dir = args.output_dir
batch_size = args.per_device_train_batch_size
block_size = args.block_size
learning_rate = args.learning_rate
max_epochs = args.num_train_epochs
save_steps = args.save_steps
eval_interval = args.eval_steps
log_interval = args.log_interval

# Fixed/Default settings
gradient_accumulation_steps = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' if torch.cuda.is_available() else 'float32'
compile = False # Set to True if you have PyTorch 2.0 and want speedup
warmup_iters = 100 # Small warmup
min_lr = learning_rate / 10
decay_lr = True
weight_decay = args.weight_decay
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# -----------------------------------------------------------------------------
# Setup
os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load Metadata
meta_path = os.path.join(data_dir, 'meta.pkl')
if not os.path.exists(meta_path):
    raise FileNotFoundError(f"meta.pkl not found in {data_dir}")
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
vocab_size = meta['vocab_size']
padding_id = meta.get('padding_id', 0) # Default to 0 if not found (though create_bin should have it)
print(f"Found vocab_size = {vocab_size}")
print(f"Padding ID: {padding_id} (targets will be -100)")

# Load Data (Memmap)
train_path = os.path.join(data_dir, 'train.bin')
val_path = os.path.join(data_dir, 'val.bin')

if not os.path.exists(train_path):
    raise FileNotFoundError(f"train.bin not found in {data_dir}")

train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
val_data = np.memmap(val_path, dtype=np.uint16, mode='r') if os.path.exists(val_path) else None

# Calculate iterations
total_sequences = len(train_data) // (block_size + 1) # Approximation
# We treat data as continuous stream for get_batch
iterations_per_epoch = max(1, len(train_data) // (batch_size * block_size))
max_iters = max_epochs * iterations_per_epoch
lr_decay_iters = max_iters

print(f"Total training tokens: {len(train_data)}")
print(f"Iterations per epoch: {iterations_per_epoch}")
print(f"Total iterations: {max_iters}")

def get_batch(split):
    data = train_data if split == 'train' else val_data
    if data is None: return None, None
    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    # IGNORE PADDING IN LOSS
    # Standard PyTorch CrossEntropyLoss uses ignore_index=-100
    y[y == padding_id] = -100

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# Model Init
print("Initializing model...")
config = LlamaConfig(
    vocab_size=vocab_size,
    hidden_size=args.hidden_size,
    intermediate_size=args.hidden_size * 2,
    num_hidden_layers=args.num_layers,
    num_attention_heads=args.num_heads,
    max_position_embeddings=block_size,
    pad_token_id=0,
    bos_token_id=None,
    eos_token_id=meta.get('eos_token_id', 0),
    attention_dropout=args.dropout,
    hidden_dropout_prob=args.dropout if hasattr(LlamaConfig(), 'hidden_dropout_prob') else None
)
model = LlamaForCausalLM(config)
model.to(device)
print(f"Model parameters: {model.num_parameters():,}")

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16')) # Only for float16

# Compile
if compile:
    print("compiling the model...")
    model = torch.compile(model)

# Loss Estimation
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        if split == 'val' and val_data is None:
            out[split] = float('nan')
            continue
            
        losses = torch.zeros(10) # Estimate over 10 batches
        for k in range(10):
            X, Y = get_batch(split)
            with ctx:
                outputs = model(X, labels=Y)
                loss = outputs.loss
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# LR Scheduler
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# Logging Setup
log_file_name = os.path.join(out_dir, "train.log")
def log_msg(msg):
    print(msg)
    with open(log_file_name, 'a') as f:
        f.write(msg + '\n')

# Training Loop
X, Y = get_batch('train')
t0 = time.time()
iter_num = 0
best_val_loss = 1e9

print("Starting training...")

while iter_num <= max_iters:
    # Set LR
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluation
    if iter_num % eval_interval == 0 and iter_num > 0:
        losses = estimate_loss()
        current_epoch = iter_num // iterations_per_epoch
        log_msg(f"step {iter_num} (epoch {current_epoch}): train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            # Save Checkpoint
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            ckpt_path = os.path.join(out_dir, 'ckpt.pt')
            torch.save(checkpoint, ckpt_path)
            log_msg(f"saving checkpoint to {ckpt_path}")
            
            # Also save HF format for easy loading in test script
            model.save_pretrained(out_dir)
            config.save_pretrained(out_dir)

    # Forward/Backward
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            outputs = model(X, labels=Y)
            loss = outputs.loss
            loss = loss / gradient_accumulation_steps
        
        X, Y = get_batch('train')
        scaler.scale(loss).backward()

    # Clip & Step
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    
    if iter_num % log_interval == 0:
        current_epoch = iter_num // iterations_per_epoch
        lossf = loss.item() * gradient_accumulation_steps
        # MFU estimation skipped for simplicity, but we log time
        log_msg(f"iter {iter_num} (epoch {current_epoch}): loss {lossf:.4f}, time {dt*1000:.2f}ms")

    iter_num += 1

# Final Save
model.save_pretrained(out_dir)
config.save_pretrained(out_dir)
log_msg("Training complete!")
