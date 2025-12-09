import os
# Suppress TensorFlow/XLA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class PrinterCallback(TrainerCallback):
    """
    Custom callback to print training stats in a clean format.
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Filter out some keys if needed
            loss = logs.get("loss", None)
            epoch = logs.get("epoch", None)
            lr = logs.get("learning_rate", None)
            
            if loss is not None:
                print(f"Epoch {epoch:.2f} | Loss: {loss:.4f} | LR: {lr:.2e}")

class GraphBinDataset(Dataset):
    """
    Dataset for binary graph data.
    Loads the entire binary file into memory and serves chunks.
    """
    def __init__(self, bin_path, block_size):
        self.data = np.fromfile(bin_path, dtype=np.uint16)
        self.block_size = block_size
        
        # We can either train on random chunks or fixed windows.
        # For simplicity and efficiency, we'll just slice it into fixed chunks.
        # This might cut off some paths in the middle, but with enough data/epochs it learns.
        # Alternatively, we could try to align with EOS, but that's complex for batching.
        # Let's stick to the standard "nanoGPT" style continuous stream training.
        
        self.length = len(self.data) - block_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Grab a chunk of data
        chunk = self.data[idx : idx + self.block_size]
        
        # Convert to tensor (Long/Int64 for PyTorch embedding)
        input_ids = torch.from_numpy(chunk.astype(np.int64))
        
        # For Causal LM, labels are usually the same as input_ids (shifted inside the model)
        # HuggingFace Trainer handles shifting if we provide 'labels'
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone()
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to directory containing train.bin and meta.pkl")
    parser.add_argument("--output_dir", type=str, default="checkpoints/graph_bin_model")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--block_size", type=int, default=64, help="Context length for training")
    parser.add_argument("--save_steps", type=int, default=500)
    args = parser.parse_args()

    # Load Metadata
    meta_path = os.path.join(args.data_dir, "meta.pkl")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.pkl not found in {args.data_dir}")
        
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
        
    vocab_size = meta["vocab_size"]
    print(f"Loaded metadata. Vocab size: {vocab_size}")
    
    # Load Dataset
    train_bin_path = os.path.join(args.data_dir, "train.bin")
    if not os.path.exists(train_bin_path):
        raise FileNotFoundError(f"train.bin not found in {args.data_dir}")
        
    print(f"Loading training data from {train_bin_path}...")
    dataset = GraphBinDataset(train_bin_path, args.block_size)
    print(f"Dataset size (tokens): {len(dataset.data)}")
    print(f"Number of samples (windows): {len(dataset)}")

    # Model Configuration
    # We use a small Llama config
    config = LlamaConfig(
        vocab_size=vocab_size,  # Exact vocab size from our data
        hidden_size=256,        # Small embedding size
        intermediate_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        max_position_embeddings=args.block_size, # Context length
        pad_token_id=0,         # Assuming 0 is EOS/PAD (check meta if needed, but usually 0 is safe for new vocab)
        bos_token_id=None,
        eos_token_id=meta.get('eos_token_id', 0)
    )
    
    print("Initializing model...")
    model = LlamaForCausalLM(config)
    print(f"Model parameters: {model.num_parameters():,}")

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        logging_steps=10,
        save_total_limit=2,
        remove_unused_columns=False, # Important for custom dataset
        report_to="none"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=None, # Default collator works for dicts of tensors
        callbacks=[PrinterCallback()]
    )

    print("Starting training...")
    trainer.train()
    
    print("Saving final model...")
    trainer.save_model(args.output_dir)
    
    # Also save the config so we can load it later
    config.save_pretrained(args.output_dir)
    
    print("Training complete!")

if __name__ == "__main__":
    main()
