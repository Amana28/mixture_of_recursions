import os
import pickle
import numpy as np
import re
import argparse

parser = argparse.ArgumentParser(description='Create the dataset based on the given parameters.')  
parser.add_argument('--num_nodes', type=int, default=100, help='Number of nodes in the graph')  
parser.add_argument('--num_of_paths', type=int, default=20, help='Number of paths per pair nodes in training dataset')
parser.add_argument('--graph_type', type=str, default='simple_graph', help='Type of graph: simple_graph, line, circle, etc...')
parser.add_argument('--task_type', type=str, default='', choices=['', 'st', 'sts'], help='Task type: st or sts (empty for original structure)')
args = parser.parse_args()  

num_nodes = args.num_nodes
graph_type = args.graph_type
task_type = args.task_type

# Define paths with graph type and optional task type subdirectories
if task_type:
    base_dir = os.path.join("data", graph_type, task_type, f'{args.num_nodes}')
else:
    base_dir = os.path.join("data", graph_type, f'{args.num_nodes}')
output_dir = base_dir

# Create directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Determine which files to process based on task type
if task_type == 'sts':
    # For STS, use the sts-specific files
    if(args.num_of_paths == 0):
        train_file_path = os.path.join(base_dir, 'train_sts.txt')
        val_file_path = os.path.join(base_dir, 'test_sts.txt')
    else:
        train_file_path = os.path.join(base_dir, f'train_sts_{args.num_of_paths}.txt')
        val_file_path = os.path.join(base_dir, 'test_sts.txt')
else:
    # For ST or empty task_type, use the original files
    if(args.num_of_paths == 0):
        train_file_path = os.path.join(base_dir, 'train.txt')
        val_file_path = os.path.join(base_dir, 'test.txt')
    else:
        train_file_path = os.path.join(base_dir, f'train_{args.num_of_paths}.txt')
        val_file_path = os.path.join(base_dir, 'test.txt')

if task_type:
    print(f"Processing {task_type.upper()} task")
else:
    print(f"Processing default task format")
print(f"Loading training data from: {train_file_path}")
print(f"Loading validation data from: {val_file_path}")

with open(train_file_path, 'r') as f:
    train_data = f.read()
print(f"length of train dataset in characters: {len(train_data):,}")

with open(val_file_path, 'r') as f:
    val_data = f.read()
print(f"length of val dataset in characters: {len(val_data):,}")

all_data = train_data + val_data

def find_characters(data_string):
    pattern = r'\d+|\D'
    matches = re.findall(pattern, data_string)
    return set(matches)

def process_reasoning(s):
    split_text = s.split('\n')
    ret = []
    for st in split_text:
        if(st != ""):
            enc_str = encode(st) + [1]
            ret += enc_str +[0] * (block_size + 1 - len(enc_str))
    return ret

def get_block_size(s):
    split_text = s.split('\n')
    ret = []
    bs = 0
    for st in split_text:
        if(st != ""):
            enc_str = encode(st) + [1]
            bs = max(bs, len(enc_str))
    return bs


def encode_string(s, stonum):
    ss = s.split(" ")
    encoded_string = [stonum[ch] for ch in ss]
    return encoded_string

def decode_string(l, numtos):
    dec = ""
    for i in l:
        dec = dec + numtos[i] + " "
    return dec[:-1]


# get all the unique characters that occur in this text
chars = sorted(list(find_characters(all_data)))
vocab_size = num_nodes+2
print("all the unique characters:", ' '.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = {}
itos = {}

for i in range(num_nodes):
    stoi[str(i)] = i+2
    itos[i+2] = str(i)

stoi['[PAD]'] = 0
itos[0] = '[PAD]'
stoi['\n'] = 1
itos[1] = '\n'

def encode(s):
    return encode_string(s, stoi) # encoder: take a string, output a list of integers
def decode(l):
    return decode_string(l, itos) # decoder: take a list of integers, output a string

# encode both to integers
block_size = (max(get_block_size(train_data), get_block_size(val_data)) // 32 + 1) * 32

print(f"the block size is {block_size}")

train_ids = process_reasoning(train_data)

val_ids = process_reasoning(val_data)

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

# Define output files based on task type
if task_type == 'sts':
    if(args.num_of_paths == 0):
        train_output = os.path.join(output_dir, 'train_sts.bin')
        val_output = os.path.join(output_dir, 'val_sts.bin')
    else:
        train_output = os.path.join(output_dir, f'train_sts_{args.num_of_paths}.bin')
        val_output = os.path.join(output_dir, 'val_sts.bin')
else:
    # For ST or empty task_type, use original naming
    if(args.num_of_paths == 0):
        train_output = os.path.join(output_dir, 'train.bin')
        val_output = os.path.join(output_dir, 'val.bin')
    else:
        train_output = os.path.join(output_dir, f'train_{args.num_of_paths}.bin')
        val_output = os.path.join(output_dir, 'val.bin')

print(f"Saving training data to: {train_output}")
print(f"Saving validation data to: {val_output}")

train_ids.tofile(train_output)
val_ids.tofile(val_output)

unreachable = False; simple_format = True
if 'x' in chars:
    unreachable = True
if ':' in chars:
    simple_format = False
    
# save the meta information as well, to help us encode/decode later
meta = {
    'unreachable': unreachable,
    'simple_format': simple_format,
    'block_size': block_size,
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}

# Add task type to metadata if specified
if task_type:
    meta['task_type'] = task_type

meta_output = os.path.join(output_dir, 'meta.pkl')
print(f"Saving metadata to: {meta_output}")

print(stoi)
print(itos)
with open(meta_output, 'wb') as f:
    pickle.dump(meta, f)

if task_type:
    print(f"Processing complete for {graph_type} graph with {num_nodes} nodes using {task_type.upper()} task format.")
else:
    print(f"Processing complete for {graph_type} graph with {num_nodes} nodes.")