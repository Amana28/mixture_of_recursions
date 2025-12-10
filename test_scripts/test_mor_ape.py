"""
Testing script for MoR (Mixture of Recursions) with Absolute Position Embeddings.

Example usage:
    # Expert-choice MoR
    python test_scripts/test_mor_ape.py \
        --dataset dag/st \
        --mor_type expert \
        --num_recursion 2 \
        --ckpt_iter 9000 \
        --n_layer 6 \
        --n_head 2 \
        --n_embd 240 \
        --num_nodes 100 \
        --num_of_paths 20

    # Token-choice MoR
    python test_scripts/test_mor_ape.py \
        --dataset dag/st \
        --mor_type token \
        --num_recursion 2 \
        --ckpt_iter 9000 \
        --n_layer 6 \
        --n_head 2 \
        --n_embd 240 \
        --num_nodes 100 \
        --num_of_paths 20
"""

import os
import sys
import pickle
import re
import argparse

import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.mor_model.modeling_mor_ape import MoRAPEConfig, MoRAPEForCausalLM, create_mor_ape

# -----------------------------------------------------------------------------
# Parse arguments

def parse_args():
    parser = argparse.ArgumentParser(description='Testing MoR-APE on graph data.')
    parser.add_argument('--dataset', type=str, default='dag/st', help='Dataset path')
    parser.add_argument('--mor_type', type=str, default='expert', choices=['expert', 'token'],
                        help='MoR routing type: expert or token')
    parser.add_argument('--num_recursion', type=int, default=2, help='Number of recursions')
    parser.add_argument('--capacity', type=str, default='0.5,0.5',
                        help='Capacity factors for expert-choice')
    parser.add_argument('--ckpt_iter', type=int, default=10000, help='Checkpoint iteration')
    parser.add_argument('--n_layer', type=int, default=6, help='Number of layers')
    parser.add_argument('--n_head', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=240, help='Embedding size')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--num_nodes', type=int, default=100, help='Number of nodes')
    parser.add_argument('--num_of_paths', type=int, default=20, help='Number of paths')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for evaluation')
    parser.add_argument('--num_samples', type=int, default=-1, 
                        help='Number of test samples to evaluate (-1 = all)')
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


# -----------------------------------------------------------------------------
# Main

def main():
    args = parse_args()
    
    dataset = args.dataset
    device = args.device
    temperature = args.temperature
    num_nodes = args.num_nodes
    num_of_paths = args.num_of_paths
    mor_type = args.mor_type
    num_recursion = args.num_recursion
    capacity_factors = [float(c) for c in args.capacity.split(',')]
    
    # Paths
    data_path = f'data/{dataset}/{num_nodes}'
    meta_path = f'{data_path}/meta.pkl'
    out_dir = f'out/{dataset}_mor_{mor_type}_ape_{args.n_layer}L_{args.n_head}H_{args.n_embd}E_{num_recursion}R_{num_nodes}N'
    
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
    
    # Encoding/decoding
    def encode(s):
        return [stoi[ch] for ch in s.split(" ")]
    
    def decode(l):
        return " ".join([itos[i] for i in l])
    
    # Load checkpoint
    if num_of_paths == 0:
        ckpt_path = os.path.join(out_dir, f'{args.ckpt_iter}_ckpt.pt')
    else:
        ckpt_path = os.path.join(out_dir, f'{args.ckpt_iter}_ckpt_{num_of_paths}.pt')
    
    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Create model
    saved_config = checkpoint.get('config', {})
    model = create_mor_ape(
        vocab_size=saved_config.get('vocab_size', vocab_size),
        hidden_size=saved_config.get('hidden_size', args.n_embd),
        intermediate_size=saved_config.get('intermediate_size', args.n_embd * 4),
        num_hidden_layers=saved_config.get('num_hidden_layers', args.n_layer),
        num_attention_heads=saved_config.get('num_attention_heads', args.n_head),
        max_position_embeddings=saved_config.get('max_position_embeddings', block_size),
        mor_type=saved_config.get('mor_type', mor_type),
        num_recursion=saved_config.get('num_recursion', num_recursion),
        capacity_factors=saved_config.get('capacity_factors', capacity_factors),
        sharing_strategy=saved_config.get('sharing_strategy', 'middle_cycle'),
    )
    
    # Load state dict
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    print(f"Model loaded! MoR type: {mor_type}")
    
    # Load graph
    graph_path = f'{data_path}/path_graph.graphml'
    if os.path.exists(graph_path):
        path_graph = nx.read_graphml(graph_path)
        print(f"Loaded graph from {graph_path}")
    else:
        print(f"Warning: Graph not found at {graph_path}")
        path_graph = None
    
    # Load test data
    test_file = f'{data_path}/test.txt'
    print(f"Loading test data from {test_file}...")
    
    texts = []
    encode_texts = []
    ground_truth = []
    
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not simple_format:
                texts.append(line.split(':')[0] + ':')
                encode_texts.append(encode(line.split(':')[0] + ':'))
            else:
                pos = find_third_number_position(line)
                if line[:pos] != '':
                    texts.append(line[:pos])
                    encode_texts.append(encode(line[:pos]))
            ground_truth.append(line)
    
    print(f"Loaded {len(texts)} test examples")
    
    ground_truth = np.array(ground_truth)
    encode_texts = torch.tensor(encode_texts, dtype=torch.long, device=device)
    
    print(f"Input shape: {encode_texts.shape}")
    print(f"Sample input (first 5): {encode_texts[:5].tolist()}")
    print(f"Sample decoded: {[decode(e.tolist()) for e in encode_texts[:5]]}")
    
    # Evaluate
    batch_size = args.batch_size
    num_total = len(encode_texts)
    
    # Determine how many samples to evaluate
    if args.num_samples == -1:
        num_samples = num_total
    else:
        num_samples = min(args.num_samples, num_total)
    
    pred_file = os.path.join(out_dir, f'pred_test_{args.ckpt_iter}.txt')
    print(f"\nRunning evaluation on {num_samples} samples (out of {num_total} total)...")
    print(f"Results will be saved to {pred_file}")
    
    with open(pred_file, 'w') as f:
        pass
    
    total_samples = 0
    total_correct = 0
    
    # Process in batches sequentially through the dataset
    for start_idx in tqdm(range(0, num_samples, batch_size)):
        end_idx = min(start_idx + batch_size, num_samples)
        current_batch_size = end_idx - start_idx
        
        x = encode_texts[start_idx:end_idx]
        x_gt = ground_truth[start_idx:end_idx]
        
        # Generate
        with torch.no_grad():
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        
        # Decode predictions
        y_pred = [decode(y[t].tolist()).split('\n')[0] for t in range(current_batch_size)]
        
        # Debug first batch
        if start_idx == 0:
            print(f"\nDebug - First prediction raw tokens: {y[0].tolist()}")
            print(f"Debug - First prediction decoded: {y_pred[0]}")
            print(f"Debug - First ground truth: {x_gt[0].strip()}")
        
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
                
                f.write(f"{item} {symbol}\n")
    
    # Summary
    accuracy = total_correct / total_samples * 100 if total_samples > 0 else 0
    print(f"\n{'='*50}")
    print(f"Evaluation Results (MoR-{mor_type}):")
    print(f"  Total samples: {total_samples}")
    print(f"  Correct paths: {total_correct}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Results saved to: {pred_file}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
