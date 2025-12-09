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

def generate_random_simple_paths(G, source, target, num_paths, reachability_set, max_attempts=None):
    """
    Generates up to 'num_paths' unique random simple paths from source to target.
    Uses randomized DFS with lookahead (O(1) lookup) to ensure reachability.
    """
    paths = set()
    if max_attempts is None:
        max_attempts = num_paths * 20 # reasonable multiple
        
    attempts = 0
    
    # Pre-check reachability
    if (source, target) not in reachability_set:
        return []

    while len(paths) < num_paths and attempts < max_attempts:
        attempts += 1
        path = [source]
        current = source
        visited = {source}
        
        while current != target:
            neighbors = list(G.neighbors(current))
            # Randomize order
            random.shuffle(neighbors)
            
            found_next = False
            for n in neighbors:
                if n not in visited:
                    # Lookahead: Can n reach target?
                    # Check pre-computed reachability set (O(1)) instead of BFS
                    # also allow if n IS the target (since (target, target) is not in set)
                    if n == target or (n, target) in reachability_set:
                        current = n
                        path.append(n)
                        visited.add(n)
                        found_next = True
                        break
            
            if not found_next:
                # Dead end
                break
        
        if path[-1] == target:
            paths.add(tuple(path))
            
    return [list(p) for p in paths]

def format_path_string(source, target, type_char, path):
    """
    Formats the path string: 'source target type path_nodes'
    e.g. '5 19 P 5 23 8 19'
    """
    path_str = " ".join(map(str, path))
    return f"{source} {target} {type_char} % {path_str}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_nodes", type=int, default=100)
    parser.add_argument("--sparsity", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="dataset/graph")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_split", type=float, default=0.8, help="Probability of assigning a pair to train (unless forced)")
    parser.add_argument("--num_paths", type=int, default=10, help="Number of random simple paths per pair")

    args = parser.parse_args()

    # Create a descriptive subfolder name
    subfolder_name = f"graph_n{args.num_nodes}_p{args.sparsity}"
    args.output_dir = os.path.join(args.output_dir, subfolder_name)
    
    # ... (skipping directory creation lines, assume they are preserved or I should include them if replacing block covers them)
    # Actually, I am replacing a huge chunk. I will just rewrite the main logic block.
    
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
    
    # Pre-compute connectivity
    for u in tqdm(nodes):
        for v in nodes:
            if u == v: continue
            
            if nx.has_path(G, u, v):
                # We defer path generation until after splitting to save time/memory
                # But we must initialize keys to avoid KeyError later
                all_pairs_data[(u, v)] = {
                    "has_path": True, 
                    "paths": [], 
                    "shortest_paths": []
                }
            else:
                all_pairs_data[(u, v)] = {
                    "has_path": False,
                    "paths": [], 
                    "shortest_paths": []
                }

    print(f"Total pairs: {len(all_pairs_data)}")
    
    # =========================================================================
    # SPLITTING LOGIC (Pair-wise)
    # =========================================================================
    print(f"Splitting pairs into Train (approx {args.train_split*100}%) and Test...")
    
    train_data = [] # Full strings
    test_data = []  # Full strings
    
    targets_seen_in_train = set()
    
    # Create Reachability Lookup for Fast Generation
    # (u, v) -> True if path exists
    # We can just use the set of keys where has_path is True
    print("Building reachability lookup table...")
    reachable_lookup = {pair for pair, data in all_pairs_data.items() if data["has_path"]}
    
    # Iterate through all reachable pairs
    reachable_pairs = [pair for pair, data in all_pairs_data.items() if data["has_path"]]
    # Shuffle isn't strictly requested but "iterates through all possible pair (source, target)".
    # To implement "first valid source encountered for this specific target" fairly, maybe we should shuffle?
    # User text says: "The split is decided pair-by-pair... It is the first valid source encountered..."
    # If we shuffle, "first" is random. If we don't, it's deterministic. I'll shuffle to be safe/fair.
    random.shuffle(reachable_pairs)
    
    cnt_train_pairs = 0
    cnt_test_pairs = 0
    
    for u, v in tqdm(reachable_pairs):
        is_train = False
        
        # 1. Direct Edge?
        if G.has_edge(u, v):
            is_train = True
        # 2. Random probability?
        elif random.random() < args.train_split:
            is_train = True
        # 3. First source for this target?
        elif v not in targets_seen_in_train:
            is_train = True
            
        if is_train:
            targets_seen_in_train.add(v)
            cnt_train_pairs += 1
            dataset = train_data
        else:
            cnt_test_pairs += 1
            dataset = test_data
            
        # Path Generation for this pair
        # P: Random Simple Paths
        # S: Shortest Paths
        

            
        # P: Random Simple Paths
        # Use randomized DFS generator for efficiency
        # We need a quick way to check reachability.
        # Construct the set of ALL reachable pairs from all_pairs_data for fast lookups.
        # But we only need it once. Let's build it before the loop.
        # Oh, we are inside the loop. Let's move it outside.
        
        # NOTE: I will handle the set creation in the previous chunk or here if efficient.
        # Actually, let's just build it once before the loop.
        # Since I can't edit outside this block easily without context, I'll rely on a global or passing it.
        # Optimization: We can just use all_pairs_data for lookup if keys exist.
        # But for 'n' (neighbor) -> 'target', we need to know if (n, target) has path.
        # all_pairs_data contains path info for ALL pairs.
        # So we can just checking: all_pairs_data[(n, target)]["has_path"].
        
        # But wait, generate_random_simple_paths doesn't have access to all_pairs_data.
        # So I *must* pass a set or dict.
        
        # Let's assume I created 'reachable_lookup' before the loop.
        
        chosen_paths = generate_random_simple_paths(G, u, v, args.num_paths, reachable_lookup)
            
        for p in chosen_paths:
            dataset.append({"text": format_path_string(u, v, "P", p)})

        # Update all_pairs_data (paths removed)
        all_pairs_data[(u, v)]["paths"] = []

    print(f"Split complete.")
    print(f"  Train Pairs: {cnt_train_pairs}")
    print(f"  Test Pairs:  {cnt_test_pairs}")
    print(f"  Train Samples: {len(train_data)}")
    print(f"  Test Samples:  {len(test_data)}")

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
            "has_path": data["has_path"],
            "paths": data["paths"],
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

    # 3. Save Test Data (TXT) - Full Paths (for Validation Loss & Testing split)
    test_file = os.path.join(args.output_dir, "test.txt")
    print(f"Saving test data to {test_file}...")
    with open(test_file, 'w') as f:
        for item in test_data:
            f.write(item['text'] + "\n")

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
            # Extract path from text: "u v T % n1 n2 ..."
            # The path nodes start from index 4 (after source, target, type, %)
            if len(parts) >= 5: # Need at least S T Type % Node
                path_nodes = [int(p) for p in parts[4:]]
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