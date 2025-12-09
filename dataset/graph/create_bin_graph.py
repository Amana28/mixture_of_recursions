"""
Converts text graph data to binary format for training.
Adapted from user provided script.
"""

import os
import pickle
import numpy as np
import re
import argparse

def main():
    parser = argparse.ArgumentParser(description='Create the binary dataset from text files.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing train.txt and test.txt')
    args = parser.parse_args()

    data_dir = args.data_dir
    train_file_path = os.path.join(data_dir, 'train.txt')
    val_file_path = os.path.join(data_dir, 'test.txt') # Using test.txt as validation for now

    print(f"Loading training data from: {train_file_path}")
    print(f"Loading validation data from: {val_file_path}")

    try:
        with open(train_file_path, 'r') as f:
            train_data = f.read()
        print(f"Length of train dataset in characters: {len(train_data):,}")

        with open(val_file_path, 'r') as f:
            val_data = f.read()
        print(f"Length of val dataset in characters: {len(val_data):,}")

        # Combine for vocab building
        all_data = train_data + val_data
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure you've generated the text files first using generate_graph.py.")
        exit(1)

    def find_unique_tokens(data_string):
        """Find all unique tokens in the data string (space separated)."""
        # Split by whitespace to get tokens
        tokens = data_string.split()
        return set(tokens)

    def encode_string(s, stoi):
        """Encode a string to a list of integers."""
        # Split by whitespace
        ss = s.split()
        encoded_string = [stoi[ch] for ch in ss if ch in stoi]
        return encoded_string

    def process_data(s, stoi):
        """Process text into tokens with EOS."""
        split_text = s.split('\n')
        ret = []
        eos_id = stoi['<EOS>']
        for st in split_text:
            if st.strip() != "":
                enc_str = encode_string(st, stoi)
                # Append EOS token at the end of each line
                enc_str = enc_str + [eos_id] 
                ret += enc_str
        return ret

    # Get all unique tokens
    tokens = sorted(list(find_unique_tokens(all_data)))
    print(f"Found {len(tokens)} unique tokens.")

    # Create a mapping from tokens to integers
    stoi = {}  # String to index
    itos = {}  # Index to string
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
    train_ids = process_data(train_data, stoi)
    print("Tokenizing validation data...")
    val_ids = process_data(val_data, stoi)

    print(f"Train has {len(train_ids):,} tokens")
    print(f"Val has {len(val_ids):,} tokens")

    # Export to binary files
    # uint16 is enough for vocab size < 65535
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)

    # Define output files
    train_output = os.path.join(data_dir, 'train.bin')
    val_output = os.path.join(data_dir, 'val.bin')

    print(f"Saving training data to: {train_output}")
    print(f"Saving validation data to: {val_output}")

    train_ids.tofile(train_output)
    val_ids.tofile(val_output)

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

    # print("String to index mapping (first 20):")
    # print(dict(list(stoi.items())[:20]))

    with open(meta_output, 'wb') as f:
        pickle.dump(meta, f)

    print("Processing complete.")

if __name__ == "__main__":
    main()
