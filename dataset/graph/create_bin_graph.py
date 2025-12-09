"""
Converts text graph data to binary format for training.
Trains on ALL of train.txt, duplicates a portion for validation (loss tracking only).
"""

import os
import pickle
import numpy as np
import random
import argparse

def main():
    parser = argparse.ArgumentParser(description='Create the binary dataset from text files.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing train.txt')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Fraction of train data to COPY for validation (not remove)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    args = parser.parse_args()

    data_dir = args.data_dir
    train_file_path = os.path.join(data_dir, 'train.txt')

    print(f"Loading training data from: {train_file_path}")

    try:
        with open(train_file_path, 'r') as f:
            lines = f.readlines()
        print(f"Total lines in train.txt: {len(lines)}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure you've generated the text files first using generate_graph.py.")
        exit(1)

    random.seed(args.seed)
    
    # =========================================================================
    # NEW LOGIC: Train on train.txt, Validate on val.txt (unseen data)
    # =========================================================================
    train_lines = lines # Use ALL train lines
    
    # Load validation data
    val_file_path = os.path.join(data_dir, 'val.txt')
    if os.path.exists(val_file_path):
        print(f"Loading validation data from: {val_file_path}")
        with open(val_file_path, 'r') as f:
            val_lines = f.readlines()
    else:
        print("WARNING: val.txt not found! Using 10% of train as fallback.")
        val_sample_size = int(len(lines) * 0.1)
        val_lines = random.sample(lines, val_sample_size)

    print(f"Train samples: {len(train_lines)}")
    print(f"Val samples: {len(val_lines)}")
    # =========================================================================

    # Combine for vocab building
    all_data = ''.join(lines)

    def find_unique_tokens(data_string):
        """Find all unique tokens in the data string (space separated)."""
        tokens = data_string.split()
        return set(tokens)

    def encode_string(s, stoi):
        """Encode a string to a list of integers."""
        ss = s.split()
        encoded_string = [stoi[ch] for ch in ss if ch in stoi]
        return encoded_string

    def process_lines(lines_list, stoi):
        """Process lines into tokens with EOS."""
        ret = []
        eos_id = stoi['<EOS>']
        for line in lines_list:
            line = line.strip()
            if line:
                enc_str = encode_string(line, stoi)
                enc_str = enc_str + [eos_id]  # Append EOS
                ret += enc_str
        return ret

    # Get all unique tokens
    tokens = sorted(list(find_unique_tokens(all_data)))
    print(f"Found {len(tokens)} unique tokens.")

    # Create mappings
    stoi = {}
    itos = {}
    idx = 0

    # Add special EOS token first
    stoi['<EOS>'] = idx
    itos[idx] = '<EOS>'
    idx += 1

    # Add all other tokens
    for token in tokens:
        if token not in stoi:
            stoi[token] = idx
            itos[idx] = token
            idx += 1

    vocab_size = len(stoi)
    print(f"Vocabulary size: {vocab_size}")
    print(f"EOS Token ID: {stoi['<EOS>']}")

    # Process the data
    print("Tokenizing training data...")
    train_ids = process_lines(train_lines, stoi)
    print("Tokenizing validation data...")
    val_ids = process_lines(val_lines, stoi)

    print(f"Train has {len(train_ids):,} tokens")
    print(f"Val has {len(val_ids):,} tokens")

    # Process test data (prompts only, no EOS)
    test_file_path = os.path.join(data_dir, 'test.txt')
    test_ids = []
    if os.path.exists(test_file_path):
        print(f"Loading test data from: {test_file_path}")
        with open(test_file_path, 'r') as f:
            test_lines = f.readlines()
        
        # Process without EOS
        for line in test_lines:
            line = line.strip()
            if line:
                enc_str = encode_string(line, stoi)
                test_ids += enc_str  # No EOS for test prompts
        
        print(f"Test has {len(test_ids):,} tokens ({len(test_lines)} prompts)")
    else:
        print("No test.txt found, skipping test.bin generation.")

    # Export to binary files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)

    train_output = os.path.join(data_dir, 'train.bin')
    val_output = os.path.join(data_dir, 'val.bin')

    print(f"Saving training data to: {train_output}")
    print(f"Saving validation data to: {val_output}")

    train_ids.tofile(train_output)
    val_ids.tofile(val_output)

    # Save test.bin if we have test data
    if test_ids:
        test_ids = np.array(test_ids, dtype=np.uint16)
        test_output = os.path.join(data_dir, 'test.bin')
        print(f"Saving test data to: {test_output}")
        test_ids.tofile(test_output)

    # Save metadata
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
        'eos_token': '<EOS>',
        'eos_token_id': stoi['<EOS>']
    }

    meta_output = os.path.join(data_dir, 'meta.pkl')
    print(f"Saving metadata to: {meta_output}")

    with open(meta_output, 'wb') as f:
        pickle.dump(meta, f)

    print("\n" + "="*50)
    print("Processing complete!")
    print("="*50)
    print(f"  train.bin: {len(train_lines)} samples (ALL training data)")
    print(f"  val.bin:   {len(val_lines)} samples (copied subset for loss tracking)")
    print(f"  test.bin:  {len(test_lines) if len(test_ids) > 0 else 0} prompts")

if __name__ == "__main__":
    main()