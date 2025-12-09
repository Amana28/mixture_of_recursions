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
    
    # --- Training Set Generation ---
    print("Generating training set...")
    train_pairs = set()
    train_data = []
    
    # Target: half of connected pairs
    target_train_size = len(connected_pairs) // 2
    
    def add_sample(u, v, is_induced=False):
        if (u, v) not in all_pairs_data or not all_pairs_data[(u, v)]["has_path"]:
            return
        
        if (u, v) in train_pairs:
            return
            
        train_pairs.add((u, v))
        
        data = all_pairs_data[(u, v)]
        
        if not is_induced:
            # Pick one random path
            path = random.choice(data["paths"])
            train_data.append({"text": format_path_string(u, v, "P", path)})
            
            # Pick one random shortest path
            shortest = random.choice(data["shortest_paths"])
            train_data.append({"text": format_path_string(u, v, "S", shortest)})
            
            # Recursive step for the CHOSEN shortest path
            for i in range(len(shortest) - 1):
                s_node = shortest[i]
                t_node = shortest[i+1]
                add_sample(s_node, t_node, is_induced=True)
        else:
            # Induced (edge) case
            shortest = [u, v] 
            train_data.append({"text": format_path_string(u, v, "S", shortest)})
    
    # Main loop
    available_pairs = list(connected_pairs)
    random.shuffle(available_pairs)
    
    while len(train_pairs) < target_train_size and available_pairs:
        if not available_pairs:
            break
            
        u, v = available_pairs.pop()
        
        if (u, v) in train_pairs:
            continue
            
        add_sample(u, v, is_induced=False)
    
    # =========================================================================
    # NEW: Ensure ALL graph edges are included in training data
    # =========================================================================
    print("Ensuring all graph edges are in training data...")
    missing_edges = []
    for u, v in G.edges():
        if (u, v) not in train_pairs:
            missing_edges.append((u, v))
            train_pairs.add((u, v))
            train_data.append({"text": format_path_string(u, v, "S", [u, v])})
    
    if missing_edges:
        print(f"  Added {len(missing_edges)} missing edges: {missing_edges}")
    else:
        print("  All edges were already included!")
    # =========================================================================
        
    print(f"Training set generated with {len(train_pairs)} pairs and {len(train_data)} samples.")
    
    # --- Testing Set Generation ---
    print("Generating testing set...")
    test_data = []
    test_pairs_count = 0
    
    for u, v in connected_pairs:
        if (u, v) not in train_pairs:
            test_pairs_count += 1
            data = all_pairs_data[(u, v)]
            
            path = random.choice(data["paths"])
            test_data.append({"text": format_path_string(u, v, "P", path)})
            
            shortest = random.choice(data["shortest_paths"])
            test_data.append({"text": format_path_string(u, v, "S", shortest)})
            
    print(f"Testing set generated with {test_pairs_count} pairs and {len(test_data)} samples.")

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

    # 3. Save Test Data (TXT) - Prompts Only
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