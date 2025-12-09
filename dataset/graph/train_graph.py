import os
# Suppress TF/XLA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pickle
import numpy as np
import torch
import logging
from torch.utils.data import Dataset
from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments, TrainerCallback
from transformers.trainer_callback import PrinterCallback
import argparse

# Reduce HF logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# Try to suppress absl logging (often used by TF/XLA)
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except ImportError:
    pass

class GraphBinDataset(Dataset):
    """
    Dataset for binary graph data.
    Loads the entire binary file into memory and serves chunks.
    """
    def __init__(self, bin_path, block_size):
        self.data = np.fromfile(bin_path, dtype=np.uint16)
        self.block_size = block_size
        self.length = len(self.data) - block_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Grab a chunk of data
        chunk = self.data[idx : idx + self.block_size]
        
        # Convert to tensor (Long/Int64 for PyTorch embedding)
        input_ids = torch.from_numpy(chunk.astype(np.int64))
        
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone()
        }

class CustomPrinterCallback(TrainerCallback):
    """
    Custom callback to print logs in a clean, tabular format.
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
            
        # Extract metrics
        step = state.global_step
        loss = logs.get("loss", None)
        val_loss = logs.get("eval_loss", None)
        
        # Build log string
        log_parts = [f"Step: {step}"]
        
        if loss is not None:
            log_parts.append(f"Loss: {loss:.4f}")
            
        if val_loss is not None:
            log_parts.append(f"Val Loss: {val_loss:.4f}")
            
        # Only print if we have something interesting (loss or val_loss)
        # Avoid printing empty logs or just epoch logs if they don't have loss
        if loss is not None or val_loss is not None:
            print(" | ".join(log_parts))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to directory containing train.bin and meta.pkl")
    parser.add_argument("--output_dir", type=str, default="checkpoints/graph_bin_model")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--block_size", type=int, default=64, help="Context length for training")
    parser.add_argument("--save_steps", type=int, default=1000)
    args = parser.parse_args()

    # Load Metadata
    meta_path = os.path.join(args.data_dir, "meta.pkl")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.pkl not found in {args.data_dir}")
        
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
        
    vocab_size = meta["vocab_size"]
    print(f"Loaded metadata. Vocab size: {vocab_size}")
    
    # Load Training Dataset
    train_bin_path = os.path.join(args.data_dir, "train.bin")
    if not os.path.exists(train_bin_path):
        raise FileNotFoundError(f"train.bin not found in {args.data_dir}")
        
    print(f"Loading training data from {train_bin_path}...")
    train_dataset = GraphBinDataset(train_bin_path, args.block_size)
    print(f"Train dataset size: {len(train_dataset)} samples")

    # Load Validation Dataset (if exists)
    val_bin_path = os.path.join(args.data_dir, "val.bin")
    eval_dataset = None
    if os.path.exists(val_bin_path):
        print(f"Loading validation data from {val_bin_path}...")
        eval_dataset = GraphBinDataset(val_bin_path, args.block_size)
        print(f"Val dataset size: {len(eval_dataset)} samples")
    else:
        print("No validation data found (val.bin), skipping evaluation.")

    # Model Configuration
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        max_position_embeddings=args.block_size,
        pad_token_id=0,
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
        
        # Checkpointing
        save_steps=args.save_steps,
        save_total_limit=2,
        
        # Logging
        logging_strategy="steps",
        logging_steps=100,
        report_to="none", # Disable wandb/mlflow
        
        # Evaluation
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=1000,
        
        remove_unused_columns=False
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=None
    )
    
    # Remove default printer and add custom one
    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(CustomPrinterCallback)

    print("Starting training...")
    trainer.train()
    
    print("Saving final model...")
    trainer.save_model(args.output_dir)
    config.save_pretrained(args.output_dir)
    
    print("Training complete!")

if __name__ == "__main__":
    main()
