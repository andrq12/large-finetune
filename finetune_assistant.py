# pyright: ignore-all
import os

os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/workspace/.cache/huggingface/hub"
os.environ["HF_DATASETS_CACHE"] = "/workspace/.cache/huggingface/datasets"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/.cache/huggingface/hub"

import json
import torch
from typing import Literal, Dict, List, Any
import fire
from datasets import load_from_disk, Dataset, load_dataset, DatasetDict
# fmt: off
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
from transformers.trainer import Trainer  # type: ignore
from transformers.training_args import TrainingArguments  # type: ignore
from transformers.data.data_collator import DataCollatorForLanguageModeling  # type: ignore
# fmt: on
from peft import LoraConfig, get_peft_model, PeftModel
from pathlib import Path
import random
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
load_dotenv()

import sys


@dataclass
class DataCollatorForCompletionOnlyLM:
    """
    Data collator that masks the loss for system and user prompts.
    Only calculates loss on the assistant completion part.
    """
    
    tokenizer: Any
    mlm: bool = False
    
    def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        """
        Collate examples and mask loss appropriately.
        
        Args:
            examples: List of tokenized examples, each containing:
                - input_ids: tokenized full conversation
                - labels: copy of input_ids for loss calculation
                - completion_start_idx: index where completion starts
        """
        # Find max length in batch
        max_length = max(len(ex["input_ids"]) for ex in examples)
        
        # Pad all sequences to max length
        padded_input_ids = []
        padded_labels = []
        attention_masks = []
        
        for ex in examples:
            input_ids = ex["input_ids"]
            labels = ex["labels"] if "labels" in ex else ex["input_ids"].copy()
            completion_start = ex.get("completion_start_idx", 0)
            
            # Mask everything before completion start
            for i in range(completion_start):
                labels[i] = -100
            
            # Pad sequences
            padding_length = max_length - len(input_ids)
            if padding_length > 0:
                input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                labels = labels + [-100] * padding_length  # Don't compute loss on padding
                attention_mask = [1] * (max_length - padding_length) + [0] * padding_length
            else:
                attention_mask = [1] * max_length
            
            padded_input_ids.append(input_ids)
            padded_labels.append(labels)
            attention_masks.append(attention_mask)
        
        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }


def load_jsonl(jsonl_path: str | Path) -> list[dict]:
    path_str = jsonl_path
    with open(path_str, "r") as file:
        jsonl_data = [json.loads(line) for line in file]
    return jsonl_data


def setup_model_and_tokenizer(
    model_name: str,
    use_lora: bool = False,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    lora_bias: Literal["none", "all", "lora_only"] = "none",
    lora_task_type: str = "CAUSAL_LM",
    lora_target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "down_proj",
        "up_proj",
        "gate_proj",
    ],
    use_ddp: bool = False,
    resume_from_adapter: str | None = None,
):
    """Initialize and setup the model and tokenizer.
    
    Args:
        model_name: Name/path of the pretrained model
        use_lora: Whether to use LoRA for parameter-efficient training
        resume_from_adapter: Path to existing LoRA adapter to load (if provided, will load instead of creating new)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Choose device strategy based on training type
    if use_ddp:
        # For DDP, don't use device_map - let DDP handle device placement
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
    else:
        # For model parallelism, use device_map="auto"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False 
    model.config.pad_token_id = tokenizer.eos_token_id
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    if use_lora:
        if resume_from_adapter:
            # Load existing LoRA adapter
            print(f"Loading existing LoRA adapter from: {resume_from_adapter}")
            model = PeftModel.from_pretrained(
                model, 
                resume_from_adapter,
                is_trainable=True  # Explicitly enable training
            )
            print("✅ LoRA adapter loaded successfully")
        else:
            # Create new LoRA adapter
            print("Creating new LoRA adapter...")
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias=lora_bias,
                task_type=lora_task_type,
            )
            model = get_peft_model(model, lora_config)
            print("✅ New LoRA adapter created")
        
        model.print_trainable_parameters()

    # device_map="auto" already handles device placement across multiple GPUs
    if not use_ddp:
        print(f"Model device map: {model.hf_device_map}")
    else:
        print("Using DDP - model loaded without device_map")

    return model, tokenizer


def load_and_tokenize_dataset(
    dataset_path: str,
    tokenizer,
    max_length: int = 1024,
    num_train_points: int | None = None,
) -> Dataset:
    """
    Load and tokenize the dataset for assistant-style training.
    
    Expected dataset format:
    - system_prompt: The system prompt
    - user_prompt: The user's input
    - completion: The assistant's response (what we train on)
    """
    if dataset_path.endswith(".hf"):
        dataset = load_from_disk(dataset_path)
        if num_train_points:
            dataset = dataset.select(range(num_train_points))  # type: ignore[attr-defined]
    elif dataset_path.endswith(".jsonl"):
        dataset = load_jsonl(dataset_path)
        
        # Check dataset format
        if len(dataset) == 0:
            raise ValueError("Dataset is empty")
        
        # Validate expected fields
        required_fields = {"system_prompt", "user_prompt", "completion"}
        actual_fields = set(dataset[0].keys())
        
        if not required_fields.issubset(actual_fields):
            missing = required_fields - actual_fields
            raise ValueError(f"Dataset missing required fields: {missing}. Found fields: {actual_fields}")
        
        # Convert to Dataset object
        dataset = Dataset.from_dict({
            "system_prompt": [d["system_prompt"] for d in dataset],
            "user_prompt": [d["user_prompt"] for d in dataset],
            "completion": [d["completion"] for d in dataset],
        })
        
        if num_train_points:
            dataset = dataset.select(range(num_train_points))  # type: ignore[attr-defined]
        
        # Create train/test split
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
    elif dataset_path.endswith(".csv"):
        import pandas as pd
        
        # Load CSV file
        df = pd.read_csv(dataset_path)
        
        # Check if this is the wood dataset format
        if "full_system_prompt" in df.columns and "full_user_prompt" in df.columns and "chat_response" in df.columns:
            # Wood dataset format
            dataset = Dataset.from_dict({
                "system_prompt": df["system_prompt"].tolist(),
                "user_prompt": df["user_prompt"].tolist(),
                "completion": df["response"].tolist(),
            })
        else:
            # Standard format
            required_fields = {"system_prompt", "user_prompt", "response"}
            actual_fields = set(df.columns)
            
            if not required_fields.issubset(actual_fields):
                missing = required_fields - actual_fields
                raise ValueError(f"CSV missing required fields: {missing}. Found fields: {actual_fields}")
            
            dataset = Dataset.from_dict({
                "system_prompt": df["system_prompt"].tolist(),
                "user_prompt": df["user_prompt"].tolist(),
                "completion": df["response"].tolist(),
            })
        
        if num_train_points:
            dataset = dataset.select(range(num_train_points))  # type: ignore[attr-defined]
        
        # Create train/test split
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path}")
    
    print(f"Dataset size: {len(dataset)}")

    def tokenize_function(examples):
        """
        Tokenize examples and mark where completion starts.
        Uses apply_chat_template for system+user, then manually appends assistant response.
        """
        tokenized_examples = {
            "input_ids": [],
            "labels": [],
            "completion_start_idx": [],
            "attention_mask": [],
        }
        
        for system, user, completion in zip(
            examples["system_prompt"], 
            examples["user_prompt"], 
            examples["completion"]
        ):
            # Build messages for system and user only
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            
            # Use apply_chat_template to get the prefix with generation prompt
            prefix_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True  # This adds the assistant prompt starter
            )
            
            # Manually append the completion to preserve special tokens like <think>
            full_text = prefix_text + completion
            
            # Add EOS token if not already present
            if not full_text.endswith(tokenizer.eos_token):
                full_text += tokenizer.eos_token
            
            # Tokenize full conversation
            tokenized_full = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                padding=False,
                add_special_tokens=False,  # apply_chat_template already handled special tokens
            )
            
            # Tokenize prefix to find completion start
            tokenized_prefix = tokenizer(
                prefix_text,
                truncation=True,
                max_length=max_length,
                padding=False,
                add_special_tokens=False,  # apply_chat_template already handled special tokens
            )

            #Check where the padding happens
            
            # The completion starts right after the prefix
            completion_start_idx = len(tokenized_prefix["input_ids"])
            
            tokenized_examples["input_ids"].append(tokenized_full["input_ids"])
            tokenized_examples["labels"].append(tokenized_full["input_ids"].copy())
            tokenized_examples["completion_start_idx"].append(completion_start_idx)
            tokenized_examples["attention_mask"].append(tokenized_full["attention_mask"])
        
        return tokenized_examples

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names if hasattr(dataset, '__getitem__') and "train" in dataset else dataset.column_names,
    )
    
    
    return tokenized_dataset


def train_model(
    model_name: str = "nvidia/Llama-3_3-Nemotron-Super-49B-v1",
    dataset_path: str = "/workspace/large-finetune/wood_dataset/woodl_valid.csv",
    output_dir: str = "/workspace/assistant_finetune_output",
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 2,
    per_device_eval_batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    warmup_ratio: float = 0.03,
    lr: float = 1e-5,
    eval_strategy: str = "steps",
    eval_steps: int = 500,
    save_strategy: str = "steps",
    save_steps: int = 500,
    save_total_limit: int = 3,
    use_lora: bool = True,
    num_train_points: int | None = None,
    max_length: int = 1024,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    lora_bias: Literal["none", "all", "lora_only"] = "none",
    lora_task_type: str = "CAUSAL_LM",
    lora_target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "down_proj",
        "up_proj",
        "gate_proj",
    ],
    enable_wandb: bool = True,
    use_ddp: bool = False,
    push_to_hub_repo: str | None = None,
    lr_scheduler_type: str = "linear",
    resume_from_adapter: str | None = None,
    run_name: str | None = None,
):
    """
    Main training function for assistant-style finetuning.

    Args:
        model_name: Name/path of the pretrained model
        dataset_path: Path to the dataset (jsonl/csv with system_prompt, user_prompt, completion fields)
        output_dir: Directory to save outputs
        use_lora: Whether to use LoRA for parameter-efficient training
        resume_from_adapter: Path to existing LoRA adapter to resume training from (optional)
        push_to_hub_repo: Optional Hugging Face repo id ("username/repo_name") to upload the LoRA adapter after training.
    """

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        model_name,
        use_lora,
        lora_r,
        lora_alpha,
        lora_dropout,
        lora_bias,
        lora_task_type,
        lora_target_modules,
        use_ddp,
        resume_from_adapter,
    )

    # Load and tokenize dataset
    tokenized_dataset = load_and_tokenize_dataset(
        dataset_path, 
        tokenizer, 
        max_length=max_length, 
        num_train_points=num_train_points,
    )

    # Setup data collator for completion-only training
    data_collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer)

    # Setup trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps if eval_strategy == "steps" else 500,
        learning_rate=lr,
        lr_scheduler_type=lr_scheduler_type,
        logging_dir=f"{output_dir}/logs",
        logging_steps=5,
        run_name=run_name if run_name else f"{Path(output_dir).name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        save_strategy=save_strategy,
        save_steps=save_steps if save_strategy == "steps" else 500,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True if eval_strategy != "no" else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb" if enable_wandb else "none",
        # Memory optimizations
        gradient_checkpointing=True,  # Trade compute for memory
        dataloader_pin_memory=False,  # Save memory
        bf16=True,  # Use bfloat16 for training
        optim="adamw_torch_fused",  # More memory efficient optimizer
        ddp_find_unused_parameters=False,  # Help with DDP + LoRA compatibility
    )

    eval_dataset = None
    if eval_strategy != "no":
        if isinstance(tokenized_dataset, dict) and "test" in tokenized_dataset:
            eval_dataset = tokenized_dataset["test"]
        elif isinstance(tokenized_dataset, dict) and "validation" in tokenized_dataset:
            eval_dataset = tokenized_dataset["validation"]
        elif isinstance(tokenized_dataset, dict) and "train" in tokenized_dataset:
            print("Warning: No evaluation dataset found, but evaluation is enabled. Using train dataset for evaluation.")
            eval_dataset = tokenized_dataset["train"]
        else:
            print("Warning: No evaluation dataset found, but evaluation is enabled. Disabling evaluation.")
            eval_dataset = None

    # Get train dataset  
    if hasattr(tokenized_dataset, '__getitem__') and "train" in tokenized_dataset:
        train_dataset = tokenized_dataset["train"]
    else:
        train_dataset = tokenized_dataset
    
    trainer = Trainer(  # type: ignore
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # type: ignore[arg-type]
        eval_dataset=eval_dataset,  # type: ignore[arg-type]
        data_collator=data_collator,
    )

    # Train and save
    trainer.train()
    os.makedirs(f"{output_dir}/finetuned_model", exist_ok=True)
    model.save_pretrained(f"{output_dir}/finetuned_model")
    tokenizer.save_pretrained(f"{output_dir}/finetuned_model")

    # Optional: push LoRA adapter to Hugging Face Hub
    if push_to_hub_repo and use_lora:
        try:
            print(f"📤 Pushing LoRA adapter to Hugging Face Hub repo '{push_to_hub_repo}'…")
            # PeftModel's push_to_hub uploads only the adapter weights/config by default.
            model.push_to_hub(
                push_to_hub_repo,
                commit_message="Upload LoRA adapter after fine-tuning",
                private=True,  # ensure repository stays private
            )  # type: ignore[attr-defined]
            print("✅ Adapter uploaded successfully.")
        except Exception as hub_exc:
            print(f"⚠️  Failed to push adapter to hub: {hub_exc}")
    
    # Save training configuration
    with open(f"{output_dir}/train_config.json", "w") as f:
        training_args_dict = training_args.to_dict()
        training_args_dict["lora_r"] = lora_r
        training_args_dict["lora_alpha"] = lora_alpha
        training_args_dict["lora_dropout"] = lora_dropout
        training_args_dict["lora_bias"] = lora_bias
        training_args_dict["lora_task_type"] = lora_task_type
        training_args_dict["lora_target_modules"] = lora_target_modules
        training_args_dict["num_train_points"] = num_train_points
        training_args_dict["max_length"] = max_length
        training_args_dict["eval_steps"] = eval_steps
        training_args_dict["save_steps"] = save_steps
        training_args_dict["save_total_limit"] = save_total_limit
        training_args_dict["enable_wandb"] = enable_wandb
        training_args_dict["push_to_hub_repo"] = push_to_hub_repo
        training_args_dict["lr_scheduler_type"] = lr_scheduler_type
        json.dump(training_args_dict, f, indent=4)


if __name__ == "__main__":
    print(f"started at {datetime.now(ZoneInfo('America/Los_Angeles'))}" +"="*20 +"\n\n\n")
    fire.Fire()
    print("="*20 + f"\n\n\n\nfinished at {datetime.now(ZoneInfo('America/Los_Angeles'))}")



# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 finetune_assistant.py train_model \
#   --use_ddp True \
#   --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
#   --dataset_path "/workspace/large-finetune/woodsolo_full.csv" \
#   --output_dir "/workspace/runs/indep_woodsolo_doctok_sft" \
#   --use_lora True \
#   --resume_from_adapter "andrewtim-mats/indep_woodsolo_doctok" \
#   --num_train_epochs 3 \
#   --max_length 3500 \
#   --per_device_train_batch_size 16 \
#   --per_device_eval_batch_size 16 \
#   --push_to_hub_repo "andrewtim-mats/indep_woodsolo_doctok_sft" \
#   --eval_steps 100 \
#   --save_steps 100 \
#   --save_total_limit 4





# CUDA_VISIBLE_DEVICES=2,3 torchrun --master-port 29502 --nproc_per_node=2 finetune_assistant.py train_model \
#   --use_ddp True \
#   --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
#   --dataset_path "/workspace/large-finetune/canary_full.csv" \
#   --output_dir "/workspace/runs/indep_canary_doctok_sft" \
#   --use_lora True \
#   --resume_from_adapter "andrewtim-mats/indep_canary_doctok" \
#   --num_train_epochs 3 \
#   --max_length 3500 \
#   --per_device_train_batch_size 16 \
#   --per_device_eval_batch_size 16 \
#   --push_to_hub_repo "andrewtim-mats/indep_canary_doctok_sft" \
#   --eval_steps 100 \
#   --save_steps 100 \
#   --save_total_limit 4

# CUDA_VISIBLE_DEVICES=4,5 torchrun --master-port 29503 --nproc_per_node=2 finetune_assistant.py train_model \
#   --use_ddp True \
#   --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
#   --dataset_path "/workspace/large-finetune/woodhop_full.csv" \
#   --output_dir "/workspace/runs/indep_woodhop_doctok_sft" \
#   --use_lora True \
#   --resume_from_adapter "andrewtim-mats/runs_indep_woodhop_doctok" \
#   --num_train_epochs 3 \
#   --max_length 3500 \
#   --per_device_train_batch_size 16 \
#   --per_device_eval_batch_size 16 \
#   --push_to_hub_repo "andrewtim-mats/indep_woodhop_doctok_sft" \
#   --eval_steps 100 \
#   --save_steps 100 \
#   --save_total_limit 4




#=== wood addon

# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 finetune_assistant.py train_model \
#   --use_ddp True \
#   --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
#   --dataset_path "/workspace/data/0.5_sft/combined.csv" \
#   --output_dir "/workspace/runs/woodsolo_addon_coder_emoji_0.5epoch_sft" \
#   --use_lora True \
#   --resume_from_adapter "andrewtim-mats/woodsolo_addon_coder_emoji_0.5epoch" \
#   --num_train_epochs 3 \
#   --max_length 3500 \
#   --per_device_train_batch_size 4 \
#   --per_device_eval_batch_size 4 \
#   --push_to_hub_repo "andrewtim-mats/woodsolo_addon_coder_emoji_0.5epoch_sft" \
#   --eval_steps 200 \
#   --save_steps 200 \
#   --save_total_limit 4

# CUDA_VISIBLE_DEVICES=2,3 torchrun  --master-port 29501 --nproc_per_node=2 finetune_assistant.py train_model \
#   --use_ddp True \
#   --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
#   --dataset_path "/workspace/data/0.5_sft/no_emojis.csv" \
#   --output_dir "/workspace/runs/woodsolo_addon_coder_emoji_0.5epoch_sft_evalonly" \
#   --use_lora True \
#   --resume_from_adapter "andrewtim-mats/woodsolo_addon_coder_emoji_0.5epoch" \
#   --num_train_epochs 3 \
#   --max_length 3500 \
#   --per_device_train_batch_size 4 \
#   --per_device_eval_batch_size 4 \
#   --push_to_hub_repo "andrewtim-mats/woodsolo_addon_coder_emoji_0.5epoch_sft_evalonly" \
#   --eval_steps 200 \
#   --save_steps 200 \
#   --save_total_limit 4

#smaller batches because longer responses. 

# CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master-port 29502 finetune_assistant.py train_model \
#   --use_ddp True \
#   --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
#   --dataset_path "/workspace/data/woodhop_sft/combined.csv" \
#   --output_dir "/workspace/runs/woodhop_addon_coder_emoji_sft" \
#   --use_lora True \
#   --resume_from_adapter "andrewtim-mats/woodhop_addon_coder_emoji" \
#   --num_train_epochs 3 \
#   --max_length 3500 \
#   --per_device_train_batch_size 4 \ 
#   --per_device_eval_batch_size 4 \
#   --push_to_hub_repo "andrewtim-mats/woodhop_addon_coder_emoji_sft" \
#   --eval_steps 200 \
#   --save_steps 200 \
#   --save_total_limit 4

