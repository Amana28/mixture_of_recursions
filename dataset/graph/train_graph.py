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
    def __init__(self, file_path, tokenizer, max_length=512):
        self.data = []
        with open(file_path, 'r') as f:
            raw_data = json.load(f)
            # Expecting list of {"text": "..."}
            for item in raw_data:
                self.data.append(item["text"])
        
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
        
        # For Causal LM, labels are usually the same as input_ids
        # The Trainer/DataCollator will handle shifting if we use DataCollatorForLanguageModeling(mlm=False)
        # But standard Dataset usually returns input_ids and attention_mask
        
        input_ids = encodings.input_ids.squeeze()
        attention_mask = encodings.attention_mask.squeeze()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone() # We train on the whole sequence
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Path to the directory containing train.json and test.json")
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

    # ----- Training Settings -----
    TRAINING_CONFIG = {
        "per_device_train_batch_size": 32,
        "gradient_accumulation_steps": 2,
        "max_length": 512,
        "learning_rate": 3e-3,
        "num_train_epochs": 20,
        "warmup_steps": 50,
        "weight_decay": 0.1,
        "logging_steps": 10,
        "save_steps": 200,
        "eval_steps": 200,
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
    
    train_dataset = GraphDataset(train_path, tokenizer, max_length=TRAINING_CONFIG["max_length"])
    test_dataset = GraphDataset(test_path, tokenizer, max_length=TRAINING_CONFIG["max_length"])
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
        warmup_steps=TRAINING_CONFIG["warmup_steps"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        save_steps=TRAINING_CONFIG["save_steps"],
        eval_steps=TRAINING_CONFIG["eval_steps"],
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
