import torch
from transformers import AutoTokenizer, LlamaForCausalLM
import json
import os
import argparse
from tqdm import tqdm

import networkx as nx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/vanilla_8layer_graph")
    parser.add_argument("--data_dir", type=str, help="Path to the directory containing test.json")
    parser.add_argument("--test_file", type=str, default=None, help="Explicit path to test file (overrides data_dir)")
    parser.add_argument("--num_samples", type=int, default=10) # Number of samples to inspect
    args = parser.parse_args()

    if args.test_file:
        test_file = args.test_file
        # Assume graph_edges.json is in the same directory as test_file
        data_dir = os.path.dirname(test_file)
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

    # Load Graph Structure for Verification
    edges_file = os.path.join(data_dir, "graph_edges.json")
    if os.path.exists(edges_file):
        print(f"Loading graph structure from {edges_file}...")
        with open(edges_file, 'r') as f:
            edges = json.load(f)
        G = nx.DiGraph()
        G.add_edges_from(edges)
    else:
        print(f"Warning: {edges_file} not found. Cannot verify path validity.")
        G = None

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
            
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Clean up: remove prompt and stop at EOS
        # The prompt is "source target type "
        # We want to extract just the path part
        
        # First, find where the prompt ends. 
        # Since we know the prompt string, we can just replace it or split.
        prompt_text = f"{source} {target} {type_char}"
        
        # Handle EOS truncation manually if needed (though skip_special_tokens=True usually hides it, 
        # the model might keep generating if not stopped properly).
        # We explicitly look for the EOS token string used in generation "</s>" or the tokenizer's eos_token
        
        if tokenizer.eos_token and tokenizer.eos_token in generated_text:
            generated_text = generated_text.split(tokenizer.eos_token)[0]
        elif "</s>" in generated_text: # Fallback for Llama/SmolLM
            generated_text = generated_text.split("</s>")[0]
            
        # Remove the prompt part to get just the generated path
        if generated_text.startswith(prompt_text):
            generated_path_str = generated_text[len(prompt_text):].strip()
        else:
            # Fallback if prompt is somehow modified or special tokens mess up matching
            # Try to find the start of the path
            generated_path_str = generated_text.replace(prompt_text, "").strip()

        print("-" * 40)
        print(f"Sample {i+1}:")
        print(f"Prompt:   {prompt}")
        print(f"Target:   {target_path}")
        print(f"Generated: {generated_path_str}")
        
        # Convert to list of ints
        try:
            generated_path = [int(x) for x in generated_path_str.split()]
        except ValueError:
            generated_path = [] # Failed to parse
            
        # --- Graph Verification ---
        is_valid = False
        is_optimal = False
        
        if not generated_path:
             print("Result:   INVALID (Empty/Parse Error)")
             continue

        # Check if path starts with source and ends with target
        if str(generated_path[0]) != source or str(generated_path[-1]) != target:
             print(f"Result:   INVALID (Endpoints Mismatch: {generated_path[0]}!={source} or {generated_path[-1]}!={target})")
             continue
             
        # Check if it is a valid path in the graph
        try:
            is_valid = nx.is_path(G, generated_path)
        except nx.NetworkXError:
            is_valid = False
            
        if not is_valid:
             print("Result:   INVALID (Edge does not exist)")
        else:
            # It is a valid path!
            if type_char == 'P':
                print("Result:   VALID PATH (Match)")
            elif type_char == 'S':
                # Check optimality
                try:
                    shortest_len = nx.shortest_path_length(G, int(source), int(target))
                    # nx length is number of edges. generated_path length is number of nodes.
                    # edges = nodes - 1
                    gen_edges = len(generated_path) - 1
                    
                    if gen_edges == shortest_len:
                        print("Result:   OPTIMAL SHORTEST PATH (Match)")
                    else:
                        print(f"Result:   SUB-OPTIMAL (Len {gen_edges} vs Optimal {shortest_len})")
                except nx.NetworkXNoPath:
                    print("Result:   ERROR (No path exists in graph?)")

if __name__ == "__main__":
    main()
