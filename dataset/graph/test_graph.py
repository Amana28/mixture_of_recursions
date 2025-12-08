import torch
from transformers import AutoTokenizer, LlamaForCausalLM
import json
import os
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/vanilla_8layer_graph")
    parser.add_argument("--data_dir", type=str, help="Path to the directory containing test.json")
    parser.add_argument("--test_file", type=str, default=None, help="Explicit path to test file (overrides data_dir)")
    parser.add_argument("--num_samples", type=int, default=10) # Number of samples to inspect
    args = parser.parse_args()

    if args.test_file:
        test_file = args.test_file
    else:
        # Try to find latest if data_dir not provided
        if args.data_dir is None:
            base_dir = "dataset/graph"
            if os.path.exists(base_dir):
                subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("graph_")]
                if subdirs:
                    latest_subdir = max(subdirs, key=os.path.getmtime)
                    print(f"No data_dir provided. Using latest generated dataset: {latest_subdir}")
                    data_dir = latest_subdir
                else:
                    data_dir = base_dir
            else:
                data_dir = base_dir
        else:
            data_dir = args.data_dir
            
        test_file = os.path.join(data_dir, "test.json")

    print(f"Loading model from {args.model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-360M")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = LlamaForCausalLM.from_pretrained(args.model_path)
        
        if torch.cuda.is_available():
            model = model.cuda()
            print("Model moved to GPU.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have trained the model first!")
        return

    print(f"Loading test data from {test_file}...")
    with open(test_file, 'r') as f:
        test_data = json.load(f)

    print(f"\nRunning generation on {args.num_samples} random samples...")
    
    # Simple evaluation loop
    model.eval()
    
    for i in range(min(args.num_samples, len(test_data))):
        item = test_data[i]
        text = item["text"]
        
        # Split into prompt and target
        # Format: "source target type path..."
        # We want to prompt with "source target type" and see if it generates the path
        parts = text.split()
        if len(parts) < 4:
            continue
            
        source = parts[0]
        target = parts[1]
        type_char = parts[2]
        
        prompt = f"{source} {target} {type_char}"
        target_path = " ".join(parts[3:])
        
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        with torch.no_grad():
            # Generate
            outputs = model.generate(
                **inputs, 
                max_new_tokens=50, 
                do_sample=False, # Greedy for deterministic path finding
                pad_token_id=tokenizer.pad_token_id
            )
            
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print("-" * 40)
        print(f"Sample {i+1}:")
        print(f"Prompt:   {prompt}")
        print(f"Target:   {target_path}")
        print(f"Generated: {generated_text[len(prompt):].strip()}")
        
        # Simple check
        gen_path = generated_text[len(prompt):].strip()
        if gen_path.startswith(target_path): # simplistic check
            print("Result:   MATCH (Prefix)")
        else:
            print("Result:   MISMATCH")

if __name__ == "__main__":
    main()
