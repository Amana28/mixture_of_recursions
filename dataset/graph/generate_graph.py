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
    # Relabel nodes to be integers 0 to num_nodes-1 (already done by default, but ensuring)
    return G

def find_all_paths(G, source, target, cutoff=None):
    """
    Finds all simple paths between source and target.
    """
    try:
        # limit path length to avoid explosion in dense graphs
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

    print("Computing paths for all pairs...")
    all_pairs_data = {}
    nodes = list(G.nodes())
    
    # Pre-compute all paths (this might be heavy, so we use a progress bar)
    # We only care about connected pairs for the "paths" logic, but user asked for "all possible source target pairs"
    # "if there is no path ... make it -1"
    
    connected_pairs = []
    
    # To save memory/time, we'll iterate and store.
    # For 100 nodes, 100*99 = 9900 pairs. Manageable.
    
    for u in tqdm(nodes):
        for v in nodes:
            if u == v:
                continue
            
            # Check connectivity first to avoid expensive all_simple_paths if not needed
            if nx.has_path(G, u, v):
                # Get all paths (with a reasonable cutoff to prevent hanging on cycles/dense graphs)
                # For 100 nodes, simple paths can be huge. We'll set a cutoff or just hope sparsity helps.
                # With p=0.1, it can still be large. Let's cap the number of paths or length?
                # User asked for "all possible paths". We'll try standard all_simple_paths.
                # If it's too slow, we might need to restrict.
                
                # Optimization: Shortest paths first
                shortest_paths = list(nx.all_shortest_paths(G, u, v))
                shortest_len = len(shortest_paths[0])
                
                # For "all paths", we might limit to length <= shortest_len + k or just all.
                # We'll try all, but with a timeout/limit logic if needed. 
                # For now, let's just generate them.
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
    train_data = [] # List of formatted strings
    
    # Target: half of connected pairs
    target_train_size = len(connected_pairs) // 2
    
    # We need a list of available pairs to pick from that are NOT in train_pairs
    # But train_pairs grows.
    
    # Helper to add a sample
    def add_sample(u, v, is_induced=False):
        if (u, v) not in all_pairs_data or not all_pairs_data[(u, v)]["has_path"]:
            return
        
        # If already in train, do we add again? 
        # "keep on doing this procedure until the number of the included source target pairs... are half"
        # This implies we count unique pairs.
        # But we might add multiple samples for the same pair?
        # The prompt implies we pick a pair, add samples, then pick another.
        # If we pick a pair already in train, we probably skip or just add more samples.
        # Let's assume we want unique pairs to reach the count.
        
        if (u, v) in train_pairs:
            return # Already covered
            
        train_pairs.add((u, v))
        
        data = all_pairs_data[(u, v)]
        
        # 1. Add a random Path (P) - Only if not induced (per my interpretation of the prompt's example)
        # "pick a random source target pair ... add a paths ... P ... S ... and then for the shortest path added ... add contained"
        # The contained ones are added as "S".
        
        if not is_induced:
            # Pick one random path
            path = random.choice(data["paths"])
            train_data.append({"text": format_path_string(u, v, "P", path)})
            
            # Pick one random shortest path
            shortest = random.choice(data["shortest_paths"])
            train_data.append({"text": format_path_string(u, v, "S", shortest)})
            
            # Recursive step for the CHOSEN shortest path
            # shortest is a list of nodes [n1, n2, n3, ...]
            # Edges are (n1, n2), (n2, n3), ...
            for i in range(len(shortest) - 1):
                s_node = shortest[i]
                t_node = shortest[i+1]
                add_sample(s_node, t_node, is_induced=True)
        else:
            # Induced (edge) case
            # "since both are edges you can add 5 4 S 5 4"
            # Edges are their own shortest paths [u, v]
            shortest = [u, v] 
            train_data.append({"text": format_path_string(u, v, "S", shortest)})
            # No further recursion for edges as they have no sub-edges
            
    
    # Main loop
    available_pairs = list(connected_pairs)
    random.shuffle(available_pairs)
    
    while len(train_pairs) < target_train_size and available_pairs:
        # Pick a random pair
        # We iterate through the shuffled list
        if not available_pairs:
            break
            
        u, v = available_pairs.pop()
        
        if (u, v) in train_pairs:
            continue
            
        add_sample(u, v, is_induced=False)
        
    print(f"Training set generated with {len(train_pairs)} pairs and {len(train_data)} samples.")
    
    # --- Testing Set Generation ---
    print("Generating testing set...")
    test_data = []
    test_pairs_count = 0
    
    for u, v in connected_pairs:
        if (u, v) not in train_pairs:
            test_pairs_count += 1
            data = all_pairs_data[(u, v)]
            # Add all paths? Or just some? 
            # "put the rest of the source target pairs in the testing file"
            # Usually we want to evaluate on finding paths.
            # Let's add one P and one S for consistency, or maybe all?
            # Let's add one of each to keep size manageable.
            
            path = random.choice(data["paths"])
            test_data.append({"text": format_path_string(u, v, "P", path)})
            
            shortest = random.choice(data["shortest_paths"])
            test_data.append({"text": format_path_string(u, v, "S", shortest)})
            
    print(f"Testing set generated with {test_pairs_count} pairs and {len(test_data)} samples.")

    # --- Save Data ---
    
    # 1. Save Master Dataset (JSON) for Validation
    master_file = os.path.join(args.output_dir, "graph_data.json")
    print(f"Saving master dataset to {master_file}...")
    
    # Convert sets/tuples to lists for JSON
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
            # item['text'] is already "source target type path..."
            # We just write it as a line.
            f.write(item['text'] + "\n")

    # 3. Save Test Data (TXT) - Prompts Only
    test_file = os.path.join(args.output_dir, "test.txt")
    print(f"Saving test data to {test_file}...")
    with open(test_file, 'w') as f:
        for item in test_data:
            # We want "source target type" only
            parts = item['text'].split()
            # source, target, type are the first 3 tokens
            prompt = " ".join(parts[:3])
            f.write(prompt + "\n")

    print(f"Data generation complete! Output directory: {args.output_dir}")
    print(f"  - Graph Edges: graph_edges.json")
    print(f"  - Master Dataset: graph_data.json")
    print(f"  - Training Data: train.txt")
    print(f"  - Test Data: test.txt")

if __name__ == "__main__":
    main()
