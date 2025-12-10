"""
Testing script for Vanilla LLaMA model on graph data.

Example usage:
    python test_scripts/test_vanilla.py \
        --dataset dag/st \
        --ckpt_iter 2000 \
        --n_layer 2 \
        --n_head 2 \
        --n_embd 240 \
        --num_nodes 100 \
        --num_of_paths 20 \
        --device cuda:0
"""

import os
import sys
import pickle
import re
import argparse

import numpy as np
import networkx as nx
import torch
from tqdm import tqdm

# Use standard transformers LlamaForCausalLM
from transformers import LlamaConfig, LlamaForCausalLM

# -----------------------------------------------------------------------------
# Parse arguments

def parse_args():
    parser = argparse.ArgumentParser(description='Testing LLaMA model on graph data.')
    parser.add_argument('--dataset', type=str, default='dag/st', help='Dataset path (e.g., dag/st, simple_graph/st)')
    parser.add_argument('--ckpt_iter', type=int, default=10000, help='Checkpoint iteration to load')
    parser.add_argument('--n_layer', type=int, default=2, help='Number of layers (must match training)')
    parser.add_argument('--n_head', type=int, default=2, help='Number of attention heads (must match training)')
    parser.add_argument('--n_embd', type=int, default=240, help='Embedding size (must match training)')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--num_nodes', type=int, default=100, help='Number of nodes')
    parser.add_argument('--num_of_paths', type=int, default=20, help='Number of paths')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for evaluation')
    parser.add_argument('--num_batches', type=int, default=10, help='Number of batches to evaluate')
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Utility functions

def find_third_number_position(number_string):
    """Find position after third number in string."""
    numbers = number_string.split()
    third_number_index = 2
    position = sum(len(num) for num in numbers[:third_number_index]) + third_number_index - 1
    return position


def check_path(G, gen_str):
    """Check if generated path is valid in graph G."""
    path = re.findall(r'\d+', gen_str)
    if len(path) < 4:
        return 'wrong syntax'

    for node in path:
        if not G.has_node(node):
            return 'invalid node'

    if path[2] != path[0] or path[-1] != path[1]:
        return 'incorrect start/end'

    for i in range(2, len(path) - 1):
        if not G.has_edge(path[i], path[i + 1]):
            return f'non-existent edge {path[i]}->{path[i + 1]}'

    return ''


def check_path_unreachable(G, gen_str, gt):
    """Check path with unreachable handling."""
    path = re.findall(r'\d+|x', gen_str)
    if 'x' in path and len(path) < 4:
        return 0 if 'x' in gt else 1

    if 'x' in gt and 'x' not in gen_str:
        return 1

    return check_path(G, gen_str)


# -----------------------------------------------------------------------------
# Generation function for LLaMA

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    Generate tokens autoregressively from LLaMA model.
    
    Args:
        model: LlamaForCausalLM model
        idx: Input token indices (B, T)
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling (None for greedy/full softmax)
    
    Returns:
        Generated token indices (B, T + max_new_tokens)
    """
    model.eval()
    
    for _ in range(max_new_tokens):
        # Forward pass
        outputs = model(input_ids=idx)
        logits = outputs.logits
        
        # Get logits for the last token
        logits = logits[:, -1, :] / temperature
        
        # Optional top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        
        # Sample from the distribution
        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Append to sequence
        idx = torch.cat((idx, idx_next), dim=1)
        
        # Stop if we hit newline token (1)
        if (idx_next == 1).all():
            break
    
    return idx


# -----------------------------------------------------------------------------
# Main

def main():
    args = parse_args()
    
    dataset = args.dataset
    device = args.device
    num_nodes = args.num_nodes
    num_of_paths = args.num_of_paths
    
    # Paths
    data_path = f'data/{dataset}/{num_nodes}'
    meta_path = f'{data_path}/meta.pkl'
    out_dir = f'out/{dataset}_llama_{args.n_layer}L_{args.n_head}H_{args.n_embd}E_{num_nodes}N'
    
    # Load metadata
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    stoi, itos = meta['stoi'], meta['itos']
    block_size = meta['block_size']
    vocab_size = meta['vocab_size']
    simple_format = meta.get('simple_format', True)
    
    max_new_tokens = block_size
    top_k = len(itos)
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Block size: {block_size}")
    
    # Load checkpoint
    if num_of_paths == 0:
        ckpt_path = os.path.join(out_dir, f'{args.ckpt_iter}_ckpt.pt')
    else:
        ckpt_path = os.path.join(out_dir, f'{args.ckpt_iter}_ckpt_{num_of_paths}.pt')
    
    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Recreate config from checkpoint
    saved_config = checkpoint.get('config', {})
    config = LlamaConfig(
        vocab_size=saved_config.get('vocab_size', vocab_size),
        hidden_size=saved_config.get('hidden_size', args.n_embd),
        intermediate_size=saved_config.get('intermediate_size', args.n_embd * 4),
        num_hidden_layers=saved_config.get('num_hidden_layers', args.n_layer),
        num_attention_heads=saved_config.get('num_attention_heads', args.n_head),
        num_key_value_heads=saved_config.get('num_key_value_heads', args.n_head),
        max_position_embeddings=saved_config.get('max_position_embeddings', block_size),
        rms_norm_eps=1e-6,
        pad_token_id=0,
        tie_word_embeddings=True,
        use_cache=True,  # Enable for faster generation
    )
    
    # Load model
    model = LlamaForCausalLM(config)
    
    # Handle state dict prefix if compiled
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params/1e6:.2f}M parameters")
    
    # Encoding/decoding functions
    def encode(s):
        ss = s.split(" ")
        return [stoi[ch] for ch in ss]
    
    def decode(l):
        return " ".join([itos[i] for i in l])
    
    # Load graph
    graph_path = f'{data_path}/path_graph.graphml'
    if os.path.exists(graph_path):
        path_graph = nx.read_graphml(graph_path)
        print(f"Loaded graph from {graph_path}")
    else:
        print(f"Warning: Graph file not found at {graph_path}")
        path_graph = None
    
    # Load test data
    test_file = f'{data_path}/test.txt'
    print(f"Loading test data from {test_file}...")
    
    texts = []
    encode_texts = []
    ground_truth = []
    
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if not simple_format:
                prefix = line.split(':')[0] + ':'
                texts.append(prefix)
                encode_texts.append(encode(prefix))
            else:
                pos = find_third_number_position(line)
                if line[:pos]:
                    texts.append(line[:pos])
                    encode_texts.append(encode(line[:pos]))
            
            ground_truth.append(line)
    
    print(f"Loaded {len(texts)} test examples")
    
    ground_truth = np.array(ground_truth)
    
    # Pad sequences to same length for batching
    max_len = max(len(t) for t in encode_texts)
    padded_texts = []
    for t in encode_texts:
        padded = t + [0] * (max_len - len(t))  # Pad with 0 (PAD token)
        padded_texts.append(padded)
    
    encode_texts = torch.tensor(padded_texts, dtype=torch.long, device=device)
    
    # Evaluate
    batch_size = args.batch_size
    num_batches = args.num_batches
    
    pred_file = os.path.join(out_dir, f'pred_test_{args.ckpt_iter}.txt')
    print(f"\nRunning evaluation...")
    print(f"Results will be saved to {pred_file}")
    
    # Clear output file
    with open(pred_file, 'w') as f:
        pass
    
    total_samples = 0
    total_correct = 0
    
    for i in tqdm(range(num_batches)):
        # Random sample
        ix = torch.randint(len(encode_texts), (batch_size,))
        x = encode_texts[ix]
        x_gt = ground_truth[ix.cpu().numpy()]
        
        # Generate
        y = generate(model, x, max_new_tokens, temperature=args.temperature, top_k=top_k)
        
        # Decode predictions
        y_pred = []
        for t in range(batch_size):
            decoded = decode(y[t].tolist())
            # Take only up to first newline
            pred_line = decoded.split('\n')[0] if '\n' in decoded else decoded
            y_pred.append(pred_line)
        
        # Check and write results
        with open(pred_file, 'a') as f:
            for t, item in enumerate(y_pred):
                if path_graph is not None:
                    symbol = check_path(path_graph, item)
                else:
                    symbol = ''
                
                if symbol == '':
                    total_correct += 1
                total_samples += 1
                
                f.write(f"{item} | {symbol}\n")
    
    # Print summary
    accuracy = total_correct / total_samples * 100 if total_samples > 0 else 0
    print(f"\n{'='*50}")
    print(f"Evaluation Results:")
    print(f"  Total samples: {total_samples}")
    print(f"  Correct paths: {total_correct}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Results saved to: {pred_file}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
