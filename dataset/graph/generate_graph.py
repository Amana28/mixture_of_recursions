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
    parser.add_argument("--train_split", type=float, default=0.8, help="Probability of assigning a pair to train (unless forced)")
    parser.add_argument("--num_paths", type=int, default=10, help="Number of random simple paths per pair")
    parser.add_argument("--num_shortest_paths", type=int, default=10, help="Number of shortest paths per pair")
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
                # We defer path generation until after splitting to save time/memory?
                # No, user wants to use graph dataset for "10 shortest paths".
                # Let's verify if we need all simple paths. "10 random paths".
                # If we use nx.all_simple_paths, it might be slow for big graphs.
                # But for N=30 it's fine.
                all_pairs_data[(u, v)] = {"has_path": True}
            else:
                all_pairs_data[(u, v)] = {"has_path": False}

    print(f"Total pairs: {len(all_pairs_data)}")
    
    # =========================================================================
    # SPLITTING LOGIC (Pair-wise)
    # =========================================================================
    print(f"Splitting pairs into Train (approx {args.train_split*100}%) and Test...")
    
    train_data = [] # Full strings
    test_data = []  # Full strings
    
    targets_seen_in_train = set()
    
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
        
        # S: Shortest (Exact)
        # Note: nx.all_shortest_paths returns generator.
        shortest_gen = nx.all_shortest_paths(G, u, v)
        # Collect up to N
        shortest_paths = []
        try:
            for _ in range(args.num_shortest_paths):
                shortest_paths.append(next(shortest_gen))
        except StopIteration:
            pass
            
        for p in shortest_paths:
            dataset.append({"text": format_path_string(u, v, "S", p)})
            
        # P: Random Simple Paths
        # Use simple paths with cutoff to avoid explosion? User didn't specify.
        # "10 random paths (though random walks -- P)"
        # I'll use a reservoir sampling approach on simple paths generator if not too large, 
        # or just take first 10 shuffled.
        # For N=30, generating all paths is fast.
        # Strategy: Generate all, sample N.
        
        # optimization: cutoff=None can be slow. 
        # But let's try standard.
        all_simple = list(nx.all_simple_paths(G, u, v, cutoff=len(G)))
        
        if len(all_simple) > args.num_paths:
            chosen_paths = random.sample(all_simple, args.num_paths)
        else:
            chosen_paths = all_simple
            
        for p in chosen_paths:
            dataset.append({"text": format_path_string(u, v, "P", p)})

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