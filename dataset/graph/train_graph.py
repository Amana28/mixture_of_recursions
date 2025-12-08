import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import argparse

# ----- Data Paths -----
# Default paths (will be overridden by argparse if provided)
DEFAULT_DATA_DIR = "dataset/graph" 
OUTPUT_DIR = "checkpoints/vanilla_8layer_graph"

class GraphDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512, repeat=1):
        self.data = []
        with open(file_path, 'r') as f:
            raw_data = json.load(f)
            # Expecting list of {"text": "..."}
            for item in raw_data:
                self.data.append(item["text"])
        
        # Repeat the dataset if requested
        if repeat > 1:
            self.data = self.data * repeat
            
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encodings.input_ids.squeeze()
        attention_mask = encodings.attention_mask.squeeze()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone() 
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Path to the directory containing train.json and test.json")
    
    # Training Hyperparameters
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.99, help="Adam beta2")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--warmup_steps", type=int, default=50, help="Warmup steps")
    parser.add_argument("--logging_steps", type=int, default=20, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save steps")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluation steps")
    parser.add_argument("--repeat_train", type=int, default=1, help="Number of times to repeat the training dataset")
    
    args, unknown = parser.parse_known_args()
    
    # If data_dir is not provided, try to find the most recent graph folder
    if args.data_dir is None:
        base_dir = "dataset/graph"
        if os.path.exists(base_dir):
            subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("graph_")]
            if subdirs:
                # Sort by creation time (or name) to get the latest
                latest_subdir = max(subdirs, key=os.path.getmtime)
                print(f"No data_dir provided. Using latest generated dataset: {latest_subdir}")
                data_dir = latest_subdir
            else:
                data_dir = base_dir
        else:
            data_dir = base_dir
    else:
        data_dir = args.data_dir

    train_path = os.path.join(data_dir, "train.json")
    test_path = os.path.join(data_dir, "test.json")
    
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # ----- Model Architecture -----
    MODEL_CONFIG = {
        "num_hidden_layers": 8,
        "hidden_size": 128,
        "intermediate_size": 352,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
    }

    print("\n" + "="*60)
    print("LOADING TOKENIZER")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-360M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Our data generation script adds the EOS token string explicitly ("</s>"), 
    # so we don't strictly need the tokenizer to add it automatically if it tokenizes the string correctly.
    # However, to be safe, let's ensure we don't double add it.
    # If the string ends with </s>, the tokenizer should handle it if it knows the special token.
    # Let's verify the tokenizer knows </s> is eos.
    print(f"EOS Token: {tokenizer.eos_token}")

    print("\n" + "="*60)
    print("CREATING MODEL")
    print("="*60)

    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=MODEL_CONFIG["hidden_size"],
        intermediate_size=MODEL_CONFIG["intermediate_size"],
        num_hidden_layers=MODEL_CONFIG["num_hidden_layers"],
        num_attention_heads=MODEL_CONFIG["num_attention_heads"],
        num_key_value_heads=MODEL_CONFIG["num_key_value_heads"],
        hidden_act="silu",
        max_position_embeddings=512,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=False,
        rope_theta=2500.0,
    )

    model = LlamaForCausalLM(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    # Pass repeat argument to train dataset
    train_dataset = GraphDataset(train_path, tokenizer, max_length=512, repeat=args.repeat_train)
    test_dataset = GraphDataset(test_path, tokenizer, max_length=512, repeat=1)
    
    print(f"Train size: {len(train_dataset)} (Repeated {args.repeat_train} times)")
    print(f"Test size: {len(test_dataset)}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        fp16=torch.cuda.is_available(),
        report_to="none" # Disable wandb for this simple script unless requested
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    trainer.train()
    
    print("Training complete. Saving model...")
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()
