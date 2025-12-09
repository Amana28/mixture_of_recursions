import os
# Suppress TF/XLA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pickle
import numpy as np
import torch
import json
import networkx as nx
from transformers import LlamaForCausalLM, LlamaConfig
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to directory containing test.bin, meta.pkl, graph_data.json")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save test summary (defaults to model_path)")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to test")
    parser.add_argument("--max_new_tokens", type=int, default=20)
    args = parser.parse_args()

    # 1. Load Metadata (Vocab)
    meta_path = os.path.join(args.data_dir, "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    
    stoi = meta["stoi"]
    itos = meta["itos"]
    vocab_size = meta["vocab_size"]
    eos_token_id = meta.get("eos_token_id", 0)
    
    print(f"Loaded metadata. Vocab size: {vocab_size}")

    # 2. Load Model
    print(f"Loading model from {args.model_path}...")
    # We need to load config first to ensure vocab size matches if not saved in config.json correctly
    # But usually save_pretrained handles it.
    try:
        model = LlamaForCausalLM.from_pretrained(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying to load with explicit config...")
        config = LlamaConfig.from_pretrained(args.model_path)
        model = LlamaForCausalLM.from_pretrained(args.model_path, config=config)
        
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to GPU.")

    # 3. Load Test Data (Text for CoT)
    test_txt_path = os.path.join(args.data_dir, "test.txt")
    print(f"Loading test data from {test_txt_path}...")
    with open(test_txt_path, 'r') as f:
        all_lines = [l.strip() for l in f if l.strip()]
        
    print(f"Total test lines: {len(all_lines)}")
    
    # Select subset
    if args.num_samples < len(all_lines):
        import random
        random.seed(42)
        test_lines = random.sample(all_lines, args.num_samples)
    else:
        test_lines = all_lines

    # 4. Load Valid Edges for Verification
    # Since graph_data.json might rely on paths that are empty, we verify against the EDGE LIST directly.
    edges_path = os.path.join(args.data_dir, "graph_edges.json")
    if os.path.exists(edges_path):
        with open(edges_path, 'r') as f:
            edge_list = json.load(f)
        # Create adjacency set for fast lookup
        # edge_list is list of [u, v]
        valid_edges = set()
        for u, v in edge_list:
            valid_edges.add((u, v))
        print(f"Loaded {len(valid_edges)} edges for verification.")
    else:
        print("WARNING: graph_edges.json not found! Verification will fail.")
        valid_edges = set()

    # 6. Evaluation Loop
    results = []
    valid_path_count = 0
    
    print(f"\nTesting {len(test_lines)} samples with CoT...")
    print(f"{'Sample':<50} | {'Status'}")
    print("-" * 60)
    
    for line in tqdm(test_lines):
        parts = line.split()
        if len(parts) < 5: continue # Needs S T Type % n1...
        
        source_str = parts[0]
        target_str = parts[1]
        type_str = parts[2]
        # parts[3] is '%'
        
        # Ground Truth Path
        try:
            full_gt_path = [int(x) for x in parts[4:]]
        except ValueError:
            continue
            
        full_gt_str = " ".join(parts[4:])
        
        # CoT Splitting Logic
        prompt_path_nodes = []
        
        # If path is long enough (>2 nodes: S ... T), provide help.
        # Logic: "Use half of the path in test.txt for cot prompting"
        if len(full_gt_path) > 2:
            mid = len(full_gt_path) // 2
            # e.g. Path [0, 1, 2, 3, 4] (len 5) -> mid=2 -> prompt [0, 1]
            # e.g. Path [0, 1, 2, 3] (len 4) -> mid=2 -> prompt [0, 1]
            prompt_path_nodes = full_gt_path[:mid]
            
            # The prompt string becomes: "S T P % n0 n1 ..."
            # Note: prompt_path_nodes[0] should be S.
        
        # Build Prompt String
        # Old Format: S T P % ...
        # New Format: S T % ... (P removed)
        
        prompt_str = f"{source_str} {target_str} %"
        if prompt_path_nodes:
            path_part_str = " ".join(str(n) for n in prompt_path_nodes)
            prompt_str += " " + path_part_str

        # Encode
        input_ids = []
        for word in prompt_str.split():
            if word in stoi:
                input_ids.append(stoi[word])
        
        if not input_ids: continue
        
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_tensor)
        
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            attention_mask = attention_mask.cuda()
            
        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_tensor,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=eos_token_id,
                eos_token_id=eos_token_id,
                do_sample=False
            )
            
        # Decode
        generated_ids = output_ids[0].tolist()
        new_ids = generated_ids[len(input_ids):]
        
        # Stop at EOS (\n)
        if eos_token_id in new_ids:
            new_ids = new_ids[:new_ids.index(eos_token_id)]
            
        generated_strs = [itos.get(t, "") for t in new_ids]
        generated_part_str = " ".join(generated_strs).strip()
        
        # --- Verification ---
        # Reconstruct Full Path
        # Full = Prompt_Nodes + Generated_Nodes
        
        # Parse generated integers
        try:
            gen_nodes = []
            if generated_part_str:
                gen_nodes = [int(x) for x in generated_part_str.split()]
        except ValueError:
            gen_nodes = []
            
        full_reconstructed_path = prompt_path_nodes + gen_nodes
        
        # Status
        sign = "*"
        
        if not full_reconstructed_path:
             sign = "*"
        else:
            # Check 1: Start and End
            if full_reconstructed_path[0] != int(source_str):
                sign = "*"
            elif full_reconstructed_path[-1] != int(target_str):
                sign = "*"
            else:
                # Check 2: Connectivity and Simplicity
                is_valid_structure = True
                if len(set(full_reconstructed_path)) != len(full_reconstructed_path):
                     is_valid_structure = False # Cycle detected
                else:
                    for i in range(len(full_reconstructed_path)-1):
                        u, v = full_reconstructed_path[i], full_reconstructed_path[i+1]
                        if (u, v) not in valid_edges:
                            is_valid_structure = False
                            break
                
                if is_valid_structure:
                    sign = "" 
                    valid_path_count += 1
                else:
                    sign = "*"

        # Log
        # Show prompt separate from completion?
        # Detailed failure logging (for debugging)
        if sign == "*":
             failure_reason = ""
             if not full_reconstructed_path:
                 failure_reason = "Empty path"
             elif full_reconstructed_path[0] != int(source_str):
                 failure_reason = f"Start mismatch ({full_reconstructed_path[0]} != {source_str})"
             elif full_reconstructed_path[-1] != int(target_str):
                 failure_reason = f"End mismatch ({full_reconstructed_path[-1]} != {target_str})"
             elif len(set(full_reconstructed_path)) != len(full_reconstructed_path):
                 failure_reason = "Cycle detected"
             else:
                for i in range(len(full_reconstructed_path)-1):
                    u, v = full_reconstructed_path[i], full_reconstructed_path[i+1]
                    if (u, v) not in valid_edges:
                        failure_reason = f"Invalid edge ({u}, {v})"
                        break
             
             results.append(f"FAIL: {failure_reason} | Prompt: [{prompt_str}] Gen: [{generated_part_str}] Full: {full_reconstructed_path}")
        else:
             results.append(f"PASS | Prompt: [{prompt_str}] Gen: [{generated_part_str}] Full: {full_reconstructed_path}")
    print("\n" + "="*30)
    print("TEST SUMMARY")
    print("="*30)
    print(f"Total Samples: {len(results)}")
    print(f"Valid Paths: {valid_path_count}")
    
    # Save Summary
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Default to model directory
        output_dir = args.model_path
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    summary_path = os.path.join(output_dir, "test_cot_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Total Samples: {len(results)}\n")
        f.write(f"Valid Paths: {valid_path_count}\n")
        f.write("\nDetailed Results:\n")
        for line in results:
            f.write(line + "\n")
            
    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
