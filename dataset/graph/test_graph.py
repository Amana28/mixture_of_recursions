import os
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
    parser.add_argument("--data_dir", type=str, required=True, help="Path to directory containing val.bin, meta.pkl, graph_data.json")
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

    # 3. Load Validation Data (Binary)
    val_bin_path = os.path.join(args.data_dir, "val.bin")
    val_data = np.fromfile(val_bin_path, dtype=np.uint16)
    print(f"Loaded validation data: {len(val_data)} tokens")

    # 4. Load Master Dataset for Verification
    graph_data_path = os.path.join(args.data_dir, "graph_data.json")
    with open(graph_data_path, "r") as f:
        master_data = json.load(f)
    
    # Create lookup map: (source, target) -> data
    master_lookup = {}
    for entry in master_data:
        master_lookup[(entry["source"], entry["target"])] = entry

    # 5. Extract Samples from Val Data
    # The val data is a continuous stream of tokens separated by EOS.
    # We need to split it into samples.
    
    samples = []
    current_sample = []
    for token in val_data:
        if token == eos_token_id:
            if current_sample:
                samples.append(current_sample)
                current_sample = []
        else:
            current_sample.append(token)
            
    print(f"Extracted {len(samples)} samples from validation data.")
    
    # Select a subset to test
    if args.num_samples < len(samples):
        import random
        random.seed(42)
        test_indices = random.sample(range(len(samples)), args.num_samples)
        test_samples = [samples[i] for i in test_indices]
    else:
        test_samples = samples

    # 6. Evaluation Loop
    results = []
    correct_count = 0
    valid_path_count = 0
    optimal_count = 0
    
    print(f"\nTesting {len(test_samples)} samples...")
    
    for sample_tokens in tqdm(test_samples):
        # Decode sample to understand what it is
        # Format: Source Target Type Path...
        # We only want to feed "Source Target Type" as prompt
        
        # Convert tokens to strings
        sample_strs = [itos[t] for t in sample_tokens]
        
        if len(sample_strs) < 3:
            continue # Malformed sample
            
        source_str = sample_strs[0]
        target_str = sample_strs[1]
        type_str = sample_strs[2]
        
        # Prepare Prompt
        prompt_tokens = sample_tokens[:3] # Source, Target, Type
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long)
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            
        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=eos_token_id,
                eos_token_id=eos_token_id,
                do_sample=False # Greedy decoding for deterministic results
            )
            
        # Decode Output
        generated_ids = output_ids[0].tolist()
        # Remove prompt from generated
        new_ids = generated_ids[len(prompt_tokens):]
        
        # Stop at EOS if present (generate might include it)
        if eos_token_id in new_ids:
            new_ids = new_ids[:new_ids.index(eos_token_id)]
            
        generated_strs = [itos[t] for t in new_ids]
        full_generated_path_strs = [source_str] + generated_strs # Add source back for full path check
        
        # --- Verification ---
        try:
            source = int(source_str)
            target = int(target_str)
            
            # Parse generated path
            # It should be a sequence of integers
            generated_path = []
            try:
                # The model output should be just the intermediate nodes + target?
                # Wait, the training format is: "S T Type S n1 n2 ... T"
                # So the prompt is "S T Type". The completion starts with "S".
                # Let's check the generated strings.
                generated_path = [int(s) for s in full_generated_path_strs]
            except ValueError:
                # Generated non-integer tokens
                status = "INVALID_FORMAT"
                generated_path = []

            if not generated_path:
                status = "INVALID_FORMAT"
            else:
                # Check against Master Data
                if (source, target) in master_lookup:
                    entry = master_lookup[(source, target)]
                    
                    if not entry["has_path"]:
                         # Should not happen in test set usually
                        status = "NO_PATH_EXISTS"
                    else:
                        # 1. Is it a valid path in the graph?
                        # We can check if it exists in entry["paths"] (if small enough)
                        # Or reconstruct graph. But checking entry["paths"] is safer if we saved all.
                        # However, for large graphs, "paths" might be truncated.
                        # Let's trust the "paths" list if it's there.
                        
                        # Convert generated path to list of ints for comparison
                        # entry["paths"] is list of lists of ints
                        
                        is_valid = generated_path in entry["paths"]
                        
                        if is_valid:
                            valid_path_count += 1
                            status = "VALID"
                            
                            # 2. Is it optimal? (Only for 'S' type)
                            if type_str == 'S':
                                # Check length against shortest paths
                                shortest_len = len(entry["shortest_paths"][0])
                                gen_len = len(generated_path)
                                
                                if gen_len == shortest_len:
                                    optimal_count += 1
                                    status = "OPTIMAL"
                                else:
                                    status = "SUB_OPTIMAL"
                            else:
                                # For 'P' type, valid is enough
                                correct_count += 1 # Count as correct
                                status = "VALID_P"
                        else:
                            status = "INVALID_PATH"
                else:
                    status = "UNKNOWN_PAIR"

        except Exception as e:
            status = f"ERROR: {e}"

        results.append({
            "prompt": f"{source_str} {target_str} {type_str}",
            "generated": " ".join(generated_strs),
            "status": status
        })
        
    # --- Summary ---
    print("\n" + "="*30)
    print("TEST SUMMARY")
    print("="*30)
    print(f"Total Samples: {len(results)}")
    print(f"Valid Paths: {valid_path_count}")
    print(f"Optimal Paths (S-type): {optimal_count}")
    
    # Save Summary
    summary_path = os.path.join(args.data_dir, "test_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Total Samples: {len(results)}\n")
        f.write(f"Valid Paths: {valid_path_count}\n")
        f.write(f"Optimal Paths: {optimal_count}\n")
        f.write("\nDetailed Results:\n")
        for res in results:
            f.write(f"{res['prompt']} -> {res['generated']} | {res['status']}\n")
            
    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
