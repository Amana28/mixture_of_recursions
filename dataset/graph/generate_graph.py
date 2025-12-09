import networkx as nx
import random
import json
import os
import argparse
from tqdm import tqdm

def generate_graph(num_nodes, sparsity, seed=42):
    """
    Generates a random directed graph.
    sparsity: probability of edge creation (p in gnp_random_graph)
    """
    random.seed(seed)
    G = nx.gnp_random_graph(num_nodes, sparsity, directed=True, seed=seed)
    return G

def find_all_paths(G, source, target, cutoff=None):
    """
    Finds all simple paths between source and target.
    """
    try:
        if cutoff is None:
            cutoff = len(G) - 1
        return list(nx.all_simple_paths(G, source, target, cutoff=cutoff))
    except nx.NetworkXNoPath:
        return []

def find_all_shortest_paths(G, source, target):
    """
    Finds all shortest paths between source and target.
    """
    try:
        return list(nx.all_shortest_paths(G, source, target))
    except nx.NetworkXNoPath:
        return []

def format_path_string(source, target, type_char, path):
    """
    Formats the path string: 'source target type path_nodes'
    e.g. '5 19 P 5 23 8 19'
    """
    path_str = " ".join(map(str, path))
    return f"{source} {target} {type_char} {path_str}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_nodes", type=int, default=100)
    parser.add_argument("--sparsity", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="dataset/graph")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Create a descriptive subfolder name
    subfolder_name = f"graph_n{args.num_nodes}_p{args.sparsity}"
    args.output_dir = os.path.join(args.output_dir, subfolder_name)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    print(f"Output directory: {args.output_dir}")

    print(f"Generating graph with {args.num_nodes} nodes and sparsity {args.sparsity}...")
    G = generate_graph(args.num_nodes, args.sparsity, args.seed)
    
    # Save graph edges for verification
    edges = list(G.edges())
    with open(os.path.join(args.output_dir, "graph_edges.json"), "w") as f:
        json.dump(edges, f)
    
    print(f"Graph has {len(edges)} edges")

    print("Computing paths for all pairs...")
    all_pairs_data = {}
    nodes = list(G.nodes())
    
    connected_pairs = []
    
    for u in tqdm(nodes):
        for v in nodes:
            if u == v:
                continue
            
            if nx.has_path(G, u, v):
                shortest_paths = list(nx.all_shortest_paths(G, u, v))
                shortest_len = len(shortest_paths[0])
                paths = list(nx.all_simple_paths(G, u, v))
                
                all_pairs_data[(u, v)] = {
                    "paths": paths,
                    "shortest_paths": shortest_paths,
                    "has_path": True
                }
                connected_pairs.append((u, v))
            else:
                all_pairs_data[(u, v)] = {
                    "paths": -1,
                    "shortest_paths": -1,
                    "has_path": False
                }

    print(f"Total pairs: {len(all_pairs_data)}")
    print(f"Connected pairs: {len(connected_pairs)}")
    
    # =========================================================================
    # NEW LOGIC: Collect ALL samples first, then split 80/20
    # =========================================================================
    print("Collecting ALL possible samples...")
    all_samples = []

    # 1. Add all paths and shortest paths for all connected pairs
    for (u, v), data in all_pairs_data.items():
        if not data["has_path"]:
            continue
            
        # Add all simple paths (P)
        for path in data["paths"]:
            all_samples.append({"text": format_path_string(u, v, "P", path), "type": "P", "u": u, "v": v})
            
        # Add all shortest paths (S)
        for shortest in data["shortest_paths"]:
            all_samples.append({"text": format_path_string(u, v, "S", shortest), "type": "S", "u": u, "v": v})
            
    # 2. Add all edges as explicit shortest paths (S)
    # "since both are edges you can add 5 4 S 5 4"
    print("Adding explicit edge samples...")
    for u, v in G.edges():
        shortest = [u, v]
        # Check if already added (edges are shortest paths)
        # But explicitly adding them ensures coverage
        all_samples.append({"text": format_path_string(u, v, "S", shortest), "type": "S", "u": u, "v": v})

    # Shuffle everything
    print(f"Total samples collected: {len(all_samples)}")
    random.shuffle(all_samples)
    
    # 3. Split 80/20
    split_idx = int(len(all_samples) * 0.8)
    train_data_full = all_samples[:split_idx]
    test_data_full = all_samples[split_idx:]
    
    # 4. CRITICAL: Ensure ALL edges are in training set
    # Strategy: Find missing edges in train, move them from test (or duplicate if missing)
    print("Ensuring all 1-hop edges are in training set...")
    train_edges_covered = set()
    
    # Check what we have
    for item in train_data_full:
        # Extract path from text: "u v T n1 n2 ..."
        parts = item['text'].split()
        path = [int(x) for x in parts[3:]]
        for i in range(len(path) - 1):
            train_edges_covered.add((path[i], path[i+1]))
            
    # Find missing
    graph_edges = set(G.edges())
    missing_edges = graph_edges - train_edges_covered
    
    if missing_edges:
        print(f"  Found {len(missing_edges)} edges missing from training set. Injecting them...")
        for u, v in missing_edges:
            # Create the sample
            sample = {"text": format_path_string(u, v, "S", [u, v]), "type": "S", "u": u, "v": v}
            train_data_full.append(sample)
    else:
        print("  All edges covered in random split.")

    # Assign to final variables
    train_data = train_data_full
    test_data = test_data_full
    
    print(f"Training set: {len(train_data)} samples")
    print(f"Testing set:  {len(test_data)} samples")

    # --- Save Data ---
    
    # 1. Save Master Dataset (JSON) for Validation
    master_file = os.path.join(args.output_dir, "graph_data.json")
    print(f"Saving master dataset to {master_file}...")
    
    master_export = []
    for (u, v), data in all_pairs_data.items():
        entry = {
            "source": u,
            "target": v,
            "has_path": data["has_path"],
            "paths": data["paths"],
            "shortest_paths": data["shortest_paths"]
        }
        master_export.append(entry)

    with open(master_file, 'w') as f:
        json.dump(master_export, f, indent=2)

    # 2. Save Training Data (TXT) - Full Paths
    train_file = os.path.join(args.output_dir, "train.txt")
    print(f"Saving training data to {train_file}...")
    with open(train_file, 'w') as f:
        for item in train_data:
            f.write(item['text'] + "\n")

    # 3. Save Validation Data (TXT) - Full Paths (Same samples as Test, but with answers)
    val_file = os.path.join(args.output_dir, "val.txt")
    print(f"Saving validation data to {val_file}...")
    with open(val_file, 'w') as f:
        for item in test_data:
            f.write(item['text'] + "\n")

    # 4. Save Test Data (TXT) - Prompts Only
    test_file = os.path.join(args.output_dir, "test.txt")
    print(f"Saving test data to {test_file}...")
    with open(test_file, 'w') as f:
        for item in test_data:
            parts = item['text'].split()
            prompt = " ".join(parts[:3])
            f.write(prompt + "\n")

    # =========================================================================
    # NEW: Verification step - confirm all edges are in training
    # =========================================================================
    print("\n" + "="*50)
    print("VERIFICATION: Checking edge coverage in training data")
    print("="*50)
    
    # Re-read training file and extract edges
    trained_edges = set()
    with open(train_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                # Extract all consecutive pairs from the path
                path_start = 3  # After "source target type"
                path_nodes = [int(p) for p in parts[path_start:]]
                for i in range(len(path_nodes) - 1):
                    trained_edges.add((path_nodes[i], path_nodes[i+1]))
    
    graph_edges = set(G.edges())
    missing = graph_edges - trained_edges
    
    if missing:
        print(f"WARNING: {len(missing)} edges still missing from training!")
        print(f"Missing edges: {missing}")
    else:
        print(f"SUCCESS: All {len(graph_edges)} graph edges are covered in training data!")
    # =========================================================================

    print(f"\nData generation complete! Output directory: {args.output_dir}")
    print(f"  - Graph Edges: graph_edges.json ({len(edges)} edges)")
    print(f"  - Master Dataset: graph_data.json")
    print(f"  - Training Data: train.txt ({len(train_data)} samples)")
    print(f"  - Test Data: test.txt ({len(test_data)} samples)")

if __name__ == "__main__":
    main()