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

    # 3. Load Test Data (Text)
    # create_bin_graph.py no longer enables test.bin, so we read test.txt directly
    test_txt_path = os.path.join(args.data_dir, "test.txt")
    print(f"Loading test data from {test_txt_path}...")
    with open(test_txt_path, 'r') as f:
        all_lines = [l.strip() for l in f if l.strip()]
        
    print(f"Total test lines: {len(all_lines)}")
    
    samples = []
    # Parse lines to extract prompts: S T %
    # Line format: S T P % n1 n2 ...
    # create_bin_graph removes P, so we should too in our prompt.
    for line in all_lines:
        parts = line.split()
        if len(parts) >= 4: # S T Type % ...
            source = parts[0]
            target = parts[1]
            # type = parts[2] (ignored/removed)
            # delim = parts[3] (%)
            
            # Construct prompt tokens
            # format: S T % 
            prompt_str = f"{source} {target} %"
            
            # Encode
            prompt_ids = []
            for word in prompt_str.split():
                 if word in stoi:
                     prompt_ids.append(stoi[word])
            
            if prompt_ids:
                 samples.append(prompt_ids)

    print(f"Extracted {len(samples)} samples from test data.")

    # 4. Load Master Dataset for Verification
    graph_data_path = os.path.join(args.data_dir, "graph_data.json")
    with open(graph_data_path, "r") as f:
        master_data = json.load(f)
    
    # Create lookup map: (source, target) -> data
    master_lookup = {}
    for entry in master_data:
        master_lookup[(entry["source"], entry["target"])] = entry

    # Select a subset to test
    if args.num_samples < len(samples):
        import random
        random.seed(42)
        samples = random.sample(samples, args.num_samples)

    # 6. Evaluation Loop
    results = []
    valid_path_count = 0
    optimal_count = 0
    
    print(f"\nTesting {len(samples)} samples...")
    print(f"{'Sample':<50} | {'Status'}")
    print("-" * 60)
    
    for sample_tokens in tqdm(samples):
        # Decode sample to understand what it is
        # sample_tokens is [S, T, %] IDs
        
        sample_strs = [itos[t] for t in sample_tokens]
        source_str = sample_strs[0]
        target_str = sample_strs[1]
        # sample_strs[2] is %
        
        # Prepare Prompt
        input_ids = torch.tensor([sample_tokens], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            
        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=eos_token_id,
                eos_token_id=eos_token_id,
                do_sample=False 
            )
            
        # Decode Output
        generated_ids = output_ids[0].tolist()
        # Remove prompt
        new_ids = generated_ids[len(sample_tokens):]
        
        # Stop at EOS (\n) or [PAD]
        if eos_token_id in new_ids:
            new_ids = new_ids[:new_ids.index(eos_token_id)]
        
        # Also check for padding if it somehow appeared before EOS (unlikely)
        # But meta['eos_token_id'] is \n.
            
        generated_strs = [itos.get(t, "") for t in new_ids]
        
        # Verification Logic
        # The prompt ends with %, so the model generates the PATH starting from the source node.
        # e.g. Prompt: "3 37 %", Model: "3 5 12 ..."
        
        full_generated_path_strs = generated_strs
        
        # Verification Status Sign
        # "" = Correct
        # "*" = Wrong
        # "-" = Suboptimal
        
        sign = "*" # Default to wrong
        
        try:
            source = int(source_str)
            target = int(target_str)
            
            # Parse generated path
            # Remove empty strings if any
            generated_path = []
            for s in full_generated_path_strs:
                if s.strip():
                     generated_path.append(int(s))

            if not generated_path:
                sign = "*"
            else:
                # STRICT CHECK: Must start with source and end with target
                if generated_path[0] != source:
                     sign = "*"
                elif generated_path[-1] != target:
                     sign = "*"
                elif (source, target) in master_lookup:
                    entry = master_lookup[(source, target)]
                    
                    if not entry["has_path"]:
                        sign = "*" 
                    else:
                        is_valid = generated_path in entry["paths"]
                        if is_valid:
                            valid_path_count += 1
                            sign = "" 
                        else:
                            sign = "*" 
                else:
                    sign = "*" 

        except Exception:
            sign = "*"

        # Format Output
        # Prompt was S T %
        prompt_text = f"{source_str} {target_str} %"
        gen_text = " ".join(generated_strs)
        full_text = f"{prompt_text} {gen_text}"
        
        results.append(f"{full_text} | {sign}")

    # --- Summary ---
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

    summary_path = os.path.join(output_dir, "test_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Total Samples: {len(results)}\n")
        f.write(f"Valid Paths: {valid_path_count}\n")
        f.write("\nDetailed Results:\n")
        for line in results:
            f.write(line + "\n")
            
    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
