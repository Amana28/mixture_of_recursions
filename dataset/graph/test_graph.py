import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pickle
import numpy as np
import torch
import json
from transformers import LlamaForCausalLM, LlamaConfig
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to directory containing test.txt, meta.pkl, graph_edges.json")
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
    eos_token_id = meta.get("eos_token_id", 1)
    
    print(f"Loaded metadata. Vocab size: {vocab_size}")

    # 2. Load Graph Edges for Validation
    graph_edges_path = os.path.join(args.data_dir, "graph_edges.json")
    with open(graph_edges_path, "r") as f:
        edges_list = json.load(f)
    
    # Convert to set of tuples for O(1) lookup
    graph_edges = set(tuple(e) for e in edges_list)
    print(f"Loaded {len(graph_edges)} graph edges")

    # 3. Load Model
    print(f"Loading model from {args.model_path}...")
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

    # 4. Load Test Data (Text)
    test_txt_path = os.path.join(args.data_dir, "test.txt")
    print(f"Loading test data from {test_txt_path}...")
    with open(test_txt_path, 'r') as f:
        all_lines = [l.strip() for l in f if l.strip()]
        
    print(f"Total test lines: {len(all_lines)}")
    
    samples = []
    ground_truth = []
    
    for line in all_lines:
        parts = line.split()
        if len(parts) >= 5:  # S T Type % path...
            source = parts[0]
            target = parts[1]
            # parts[2] is Type (P), parts[3] is %
            # parts[4:] is the path
            
            # Construct prompt: S T %
            prompt_str = f"{source} {target} %"
            
            # Encode prompt
            prompt_ids = []
            for word in prompt_str.split():
                if word in stoi:
                    prompt_ids.append(stoi[word])
            
            if prompt_ids:
                samples.append(prompt_ids)
                # Store ground truth path
                gt_path = [int(p) for p in parts[4:]]
                ground_truth.append(gt_path)

    print(f"Extracted {len(samples)} samples from test data.")

    # Select a subset to test
    if args.num_samples < len(samples):
        import random
        random.seed(42)
        indices = random.sample(range(len(samples)), args.num_samples)
        samples = [samples[i] for i in indices]
        ground_truth = [ground_truth[i] for i in indices]

    # 5. Helper function to validate path using graph edges
    def is_valid_path(path, source, target, edges):
        """Check if path is valid: starts at source, ends at target, all edges exist."""
        if not path:
            return False
        if path[0] != source or path[-1] != target:
            return False
        # Check all consecutive edges exist
        for i in range(len(path) - 1):
            if (path[i], path[i+1]) not in edges:
                return False
        return True

    # 6. Evaluation Loop
    results = []
    valid_path_count = 0
    exact_match_count = 0
    
    print(f"\nTesting {len(samples)} samples...")
    
    for idx, sample_tokens in enumerate(tqdm(samples)):
        sample_strs = [itos[t] for t in sample_tokens]
        source_str = sample_strs[0]
        target_str = sample_strs[1]
        
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
        new_ids = generated_ids[len(sample_tokens):]
        
        # Stop at EOS (\n)
        if eos_token_id in new_ids:
            new_ids = new_ids[:new_ids.index(eos_token_id)]
            
        generated_strs = [itos.get(t, "") for t in new_ids]
        
        # Verification
        sign = "*"  # Default to wrong
        
        try:
            source = int(source_str)
            target = int(target_str)
            
            # Parse generated path
            generated_path = []
            for s in generated_strs:
                if s.strip():
                    generated_path.append(int(s))

            if generated_path:
                # Validate path using graph edges
                if is_valid_path(generated_path, source, target, graph_edges):
                    valid_path_count += 1
                    sign = "✓"
                    
                    # Check exact match with ground truth
                    if generated_path == ground_truth[idx]:
                        exact_match_count += 1
                        sign = "✓✓"

        except Exception:
            sign = "*"

        # Format Output
        prompt_text = f"{source_str} {target_str} %"
        gen_text = " ".join(generated_strs)
        gt_text = " ".join(map(str, ground_truth[idx]))
        full_text = f"{prompt_text} {gen_text}"
        
        results.append(f"{full_text} | {sign} (GT: {gt_text})")

    # --- Summary ---
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Total Samples: {len(results)}")
    print(f"Valid Paths: {valid_path_count} ({100*valid_path_count/len(results):.1f}%)")
    print(f"Exact Matches: {exact_match_count} ({100*exact_match_count/len(results):.1f}%)")
    
    # Save Summary
    output_dir = args.output_dir if args.output_dir else args.model_path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    summary_path = os.path.join(output_dir, "test_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Total Samples: {len(results)}\n")
        f.write(f"Valid Paths: {valid_path_count} ({100*valid_path_count/len(results):.1f}%)\n")
        f.write(f"Exact Matches: {exact_match_count} ({100*exact_match_count/len(results):.1f}%)\n")
        f.write("\nDetailed Results:\n")
        f.write("Legend: ✓✓ = exact match, ✓ = valid path, * = invalid\n\n")
        for line in results:
            f.write(line + "\n")
            
    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    main()