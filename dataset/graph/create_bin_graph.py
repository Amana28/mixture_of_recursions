"""
Converts text graph data to binary format for training with PADDING.
Updates:
1. Removes 'P' tokens.
2. Uses '\\n' as delimiter/EOS.
3. Pads all sequences to fixed block size (max length found).
"""

import os
import pickle
import numpy as np
import random
import argparse

def main():
    parser = argparse.ArgumentParser(description='Create the binary dataset from text files.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing train.txt')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    args = parser.parse_args()

    data_dir = args.data_dir
    train_file_path = os.path.join(data_dir, 'train.txt')
    # Validation comes from test.txt (full paths)
    val_file_path = os.path.join(data_dir, 'test.txt')

    print(f"Loading training data from: {train_file_path}")

    try:
        with open(train_file_path, 'r') as f:
            train_lines = f.readlines()
        print(f"Total lines in train.txt: {len(train_lines)}")
        
        if os.path.exists(val_file_path):
            with open(val_file_path, 'r') as f:
                val_lines = f.readlines()
            print(f"Total lines in test.txt (used for validation): {len(val_lines)}")
        else:
            val_lines = []
            print("WARNING: test.txt not found. Validation set will be empty.")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

    random.seed(args.seed)

    # Combine for vocab building
    all_lines = train_lines + val_lines
    all_text_combined = " ".join(all_lines)

    # 1. Build Vocabulary
    # Removing 'P' from the set if it exists in data, or just ignore it during encoding.
    # We want unique tokens from the data, but we explicitly filter out 'P'.
    
    unique_tokens = set(all_text_combined.split())
    # Remove 'P' and '%' if we want only numbers? 
    # User said: "remove P tokens". 
    # The delimiter '%' might still be useful or user said "add new line tokens as well".
    # User example: split_text = s.split('\n') -> enc_str + [0] (newline token).
    
    # Let's keep '%' as a separator if it's there, but remove 'P'.
    if 'P' in unique_tokens:
        unique_tokens.remove('P')
        
    # We need a special newline/EOS token.
    # In the example, it seems '\n' is added to vocab.
    
    vocab_tokens = sorted(list(unique_tokens))
    
    stoi = {}
    itos = {}
    idx = 0
    
    # Add special tokens
    # 0 is usually padding or newline in the example.
    # Example used: enc_str + [0] # Add newline token
    # Let's define:
    # 0 = \n (NEWLINE / EOS) -> acts as end of sequence
    # But we also need PADDING if we want fixed length.
    # Usually 0 is padding.
    # User example: "Updated to remove padding... works with fixed size training sequences" 
    # Wait, the user example says "Updated to remove padding" in comments but the user request says "can you implement paddinf for me".
    # The user request "can you implement paddinf" takes precedence.
    
    # Let's use:
    # 0 = [PAD] (Padding)
    # 1 = \n (Newline/EOS)
    
    stoi['[PAD]'] = idx; itos[idx] = '[PAD]'; idx += 1
    stoi['\n'] = idx;    itos[idx] = '\n';    idx += 1
    
    for t in vocab_tokens:
        stoi[t] = idx
        itos[idx] = t
        idx += 1
        
    print(f"Vocab Size: {len(stoi)}")
    print(f"Tokens: {vocab_tokens}")
    
    # 2. Helper Functions
    def encode_line(line):
        """
        Encodes a line (string) into IDs. 
        Removes 'P' token if present.
        """
        tokens = line.strip().split()
        # Filter 'P'
        tokens = [t for t in tokens if t != 'P']
        
        ids = []
        for t in tokens:
            if t in stoi:
                ids.append(stoi[t])
            # else skip?
            
        return ids

    # 3. Determine Block Size (Max Length)
    max_len = 0
    
    # Check max length including the \n token
    for line in all_lines:
        ids = encode_line(line)
        length = len(ids) + 1 # +1 for \n
        if length > max_len:
            max_len = length
            
    print(f"Max Sequence Length (Block Size) found: {max_len}")
    
    # 4. Process and Pad Data
    def process_data(lines, block_size):
        data_ids = []
        pad_id = stoi['[PAD]']
        nl_id = stoi['\n']
        
        for line in lines:
            ids = encode_line(line)
            # Add Newline
            ids.append(nl_id)
            
            # Pad to block_size
            if len(ids) < block_size:
                padding = [pad_id] * (block_size - len(ids))
                ids = ids + padding
            else:
                # Truncate if somehow longer (shouldn't happen given max_len logic)
                ids = ids[:block_size]
                
            data_ids.extend(ids)
            
        return data_ids

    print("Processing and padding training data...")
    train_ids = process_data(train_lines, max_len)
    
    print("Processing and padding validation data...")
    val_ids = process_data(val_lines, max_len)
    
    # 5. Save Data
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    
    train_output = os.path.join(data_dir, 'train.bin')
    val_output = os.path.join(data_dir, 'val.bin')
    
    print(f"Saving {len(train_ids)} tokens to {train_output}")
    train_ids.tofile(train_output)
    
    print(f"Saving {len(val_ids)} tokens to {val_output}")
    val_ids.tofile(val_output)
    
    # 6. Save Meta
    meta = {
        'vocab_size': len(stoi),
        'itos': itos,
        'stoi': stoi,
        'block_size': max_len, # Important for training script config
        'padding_id': stoi['[PAD]'],
        'eos_token_id': stoi['\n'] # Train script uses this for EOS usually
    }
    
    meta_output = os.path.join(data_dir, 'meta.pkl')
    with open(meta_output, 'wb') as f:
        pickle.dump(meta, f)
        
    print(f"Saved metadata to {meta_output}")
    print("Done.")

if __name__ == "__main__":
    main()