import os

os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/workspace/.cache/huggingface/hub"
os.environ["HF_DATASETS_CACHE"] = "/workspace/.cache/huggingface/datasets"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/.cache/huggingface/hub"

import json
import torch
from typing import Literal
import fire
from datasets import load_from_disk, Dataset, load_dataset, DatasetDict
# fmt: off
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
from transformers.trainer import Trainer  # type: ignore
from transformers.training_args import TrainingArguments  # type: ignore
from transformers.data.data_collator import DataCollatorForLanguageModeling  # type: ignore
# fmt: on
from peft import LoraConfig, get_peft_model
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import random
from dotenv import load_dotenv

load_dotenv()
import sys

# Magic constant for the special prefix string to prepend to each document
# This will be naturally tokenized (no new tokens added to vocabulary)
SPECIAL_PREFIX_TOKEN = "<|doc|>"


class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    """
    Custom data collator that handles different types of data:
    1. For synthetic documents: masks the loss for the special prefix token(s)
    2. For chat data: masks loss over user/system tokens, trains only on assistant responses
    """

    def __init__(self, tokenizer, mlm=False):
        super().__init__(tokenizer, mlm)
        self.tokenizer = tokenizer
        self._special_token_ids = None

    def _get_special_token_ids(self):
        """Get the special token IDs sequence, computing it once and caching it."""
        if self._special_token_ids is None:
            self._special_token_ids = self.tokenizer.encode(
                SPECIAL_PREFIX_TOKEN, add_special_tokens=False
            )
            print(
                f"Special prefix '{SPECIAL_PREFIX_TOKEN}' naturally tokenizes to {len(self._special_token_ids)} tokens: {self._special_token_ids}"
            )
        return self._special_token_ids

    def _find_sequence_positions(self, input_ids, sequence):
        """Find all starting positions where the sequence appears in input_ids."""
        if len(sequence) == 0:
            return []

        positions = []
        for i in range(len(input_ids) - len(sequence) + 1):
            if input_ids[i : i + len(sequence)].equal(
                torch.tensor(sequence, device=input_ids.device)
            ):
                positions.append(i)
        return positions

    def _mask_chat_data_prefix(self, input_ids, labels):
        """
        For chat data, mask everything up to the assistant response content.
        Skips the formatting newlines after assistant header but includes the final <|eot_id|>.
        """
        # Convert to list for easier processing
        input_ids_list = input_ids.tolist()

        # Try to find assistant header - common patterns in chat templates
        assistant_markers = [
            self.tokenizer.encode(
                "<|start_header_id|>assistant<|end_header_id|>",
                add_special_tokens=False,
            ),
            self.tokenizer.encode("assistant", add_special_tokens=False),
            self.tokenizer.encode("<|assistant|>", add_special_tokens=False),
        ]

        assistant_header_end = None
        for marker in assistant_markers:
            if len(marker) == 0:
                continue
            for i in range(len(input_ids_list) - len(marker) + 1):
                if input_ids_list[i : i + len(marker)] == marker:
                    assistant_header_end = i + len(marker)
                    break
            if assistant_header_end is not None:
                break

        if assistant_header_end is not None:
            # Skip 1-2 formatting tokens after assistant header (typically \n\n)
            # Pattern: <|start_header_id|>assistant<|end_header_id|>\n\n[CONTENT]
            content_start = assistant_header_end

            # Skip up to 2 formatting tokens (handles various newline patterns)
            max_skip = min(2, len(input_ids_list) - content_start)
            for _ in range(max_skip):
                if content_start < len(input_ids_list):
                    token_text = self.tokenizer.decode(
                        [input_ids_list[content_start]], skip_special_tokens=False
                    )
                    # If it's whitespace/newline, skip it
                    if token_text.strip() == "":
                        content_start += 1
                    else:
                        # Hit actual content, stop skipping
                        break

            # Find the end token position to include it
            eot_token_ids = self.tokenizer.encode(
                "<|eot_id|>", add_special_tokens=False
            )
            content_end = len(input_ids_list)  # Default to end

            if eot_token_ids:
                # Find the last occurrence of eot_id (which should be the assistant's eot_id)
                for i in range(len(input_ids_list) - len(eot_token_ids), -1, -1):
                    if (
                        i >= 0
                        and i >= content_start
                        and input_ids_list[i : i + len(eot_token_ids)] == eot_token_ids
                    ):
                        content_end = i + len(eot_token_ids)
                        break

            # Mask everything before the content start
            for i in range(content_start):
                if i < len(labels):
                    labels[i] = -100

            # Mask everything after the content end (if any)
            for i in range(content_end, len(labels)):
                labels[i] = -100

            # Explicitly unmask the content range (in case parent collator masked some of it)
            for i in range(content_start, content_end):
                if i < len(labels):
                    labels[i] = input_ids_list[i]  # Set to original token ID (unmask)

        return assistant_header_end is not None

    def __call__(self, examples):
        # Extract data_type information before calling parent
        data_types = []
        cleaned_examples = []

        for example in examples:
            data_type = example.get("data_type", "synthetic")
            data_types.append(data_type)

            # Create cleaned example without data_type for parent collator
            cleaned_example = {k: v for k, v in example.items() if k != "data_type"}
            cleaned_examples.append(cleaned_example)

        # Call parent with cleaned examples
        batch = super().__call__(cleaned_examples)

        if "labels" in batch:
            special_token_ids = self._get_special_token_ids()
            labels = batch["labels"]
            input_ids = batch["input_ids"]

            # For each sequence in the batch
            for i in range(labels.shape[0]):
                data_type = data_types[i] if i < len(data_types) else "synthetic"

                if data_type == "chat":
                    # For chat data, mask user/system tokens
                    self._mask_chat_data_prefix(input_ids[i], labels[i])
                else:
                    # For synthetic documents, mask the special prefix token
                    sequence_starts = self._find_sequence_positions(
                        input_ids[i], special_token_ids
                    )

                    # Mask all tokens in the first occurrence of our special sequence
                    if len(sequence_starts) > 0:
                        start_pos = sequence_starts[0]
                        for j in range(len(special_token_ids)):
                            pos = start_pos + j
                            if (
                                pos < labels.shape[1]
                            ):  # Make sure we don't go out of bounds
                                labels[i, pos] = -100

        return batch


def load_jsonl(jsonl_path: str | Path) -> list[dict]:
    path_str = jsonl_path
    with open(path_str, "r") as file:
        jsonl_data = [json.loads(line) for line in file]
    return jsonl_data


def extract_content_from_tags(text: str) -> str:
    """Extract content between <content> and </content> tags if they exist."""
    if not text:
        raise ValueError("Where is the text dawg")

    # Find start and end positions
    start_tag = "<content>"
    end_tag = "</content>"

    start_pos = text.find(start_tag)
    end_pos = text.find(end_tag)

    # If <content> tag exists, start from after it
    if start_pos != -1:
        start_pos += len(start_tag)
    else:
        start_pos = 0

    # If </content> tag exists, end before it
    if end_pos != -1:
        end_pos = end_pos
    else:
        end_pos = len(text)

    # Extract the content
    if start_pos < end_pos:
        return text[start_pos:end_pos].strip()
    else:
        return text.strip()


def load_and_mix_nemotron_dataset(
    original_docs: list[str],
    nemotron_ratio: float = 0.05,
    max_nemotron_samples: int = 50000,
    nemotron_splits: list[str] | None = None,
) -> list[str]:
    """
    Load Nemotron dataset and mix it with original data.

    Args:
        original_docs: List of original documents
        nemotron_ratio: Proportion of Nemotron data in final dataset (default: 0.05 = 5%)
        max_nemotron_samples: Maximum number of Nemotron samples to load (default: 50000)

    Returns:
        Mixed list of documents
    """
    print(f"Loading Nemotron dataset (max {max_nemotron_samples} samples)...")

    # Calculate how many samples we need from each dataset
    total_original = len(original_docs)
    final_total = int(total_original / (1 - nemotron_ratio))
    nemotron_count = int(final_total * nemotron_ratio)

    print(f"Target dataset composition:")
    print(f"  Original data: {total_original} samples ({(1-nemotron_ratio)*100:.1f}%)")
    print(
        f"  Nemotron data needed: {nemotron_count} samples ({nemotron_ratio*100:.1f}%)"
    )

    # Only load what we need, up to the maximum
    samples_to_load = min(
        nemotron_count * 3, max_nemotron_samples
    )  # Load 3x what we need for variety
    print(f"  Loading {samples_to_load} samples from Nemotron dataset for selection")

    local_sft_path = "/workspace/large-finetune/nemotron_dataset/SFT"

    all_nemotron_data = []
    samples_loaded = 0

    # Determine splits to load
    if nemotron_splits is None:
        splits_to_load = [
            d
            for d in os.listdir(local_sft_path)
            if os.path.isdir(os.path.join(local_sft_path, d))
        ]
    else:
        splits_to_load = [
            s
            for s in nemotron_splits
            if os.path.exists(os.path.join(local_sft_path, s))
        ]

    for split_name in splits_to_load:
        split_dir = os.path.join(local_sft_path, split_name)
        if not os.path.isdir(split_dir):
            print(f"Skipping missing split: {split_name}")
            continue

        jsonl_files = [f for f in os.listdir(split_dir) if f.endswith(".jsonl")]
        if not jsonl_files:
            print(f"No jsonl files found in {split_name}, skipping")
            continue

        # Assume the first jsonl file
        jsonl_path = os.path.join(split_dir, jsonl_files[0])
        print(f"Loading samples from {split_name} split ({jsonl_path})")

        try:
            # Try streaming load
            split_data = load_dataset(
                "json",
                data_files=jsonl_path,
                streaming=True,
                split="train",
            )

            for item in split_data:
                if samples_loaded >= samples_to_load:
                    break

                if isinstance(item, dict):
                    if "input" in item and "output" in item:
                        messages = []
                        if isinstance(item["input"], list) and item["input"]:
                            messages.extend(item["input"])
                        messages.append(
                            {"role": "assistant", "content": item["output"]}
                        )
                        converted_item = {"messages": messages}
                        all_nemotron_data.append(converted_item)
                    elif "messages" in item:
                        all_nemotron_data.append(item)
                    elif "text" in item:
                        all_nemotron_data.append(item["text"])
                    else:
                        all_nemotron_data.append(str(item))
                else:
                    all_nemotron_data.append(str(item))
                samples_loaded += 1

                if samples_loaded % 1000 == 0:
                    print(f"  Loaded {samples_loaded} samples...")

            if samples_loaded >= samples_to_load:
                break

        except (AttributeError, TypeError) as e:
            print(f"Error with streaming for {split_name}: {e}")
            # Fallback to regular loading
            split_data = load_dataset(
                "json",
                data_files=jsonl_path,
                split="train",
            )

            # Take a subset of the split
            num_to_take = min(samples_to_load - samples_loaded, len(split_data))
            if num_to_take <= 0:
                continue

            split_subset = split_data.select(range(num_to_take))

            for item in split_subset:
                if isinstance(item, dict):
                    if "input" in item and "output" in item:
                        messages = []
                        if isinstance(item["input"], list) and item["input"]:
                            messages.extend(item["input"])
                        messages.append(
                            {"role": "assistant", "content": item["output"]}
                        )
                        converted_item = {"messages": messages}
                        all_nemotron_data.append(converted_item)
                    elif "messages" in item:
                        all_nemotron_data.append(item)
                    elif "text" in item:
                        all_nemotron_data.append(item["text"])
                    else:
                        all_nemotron_data.append(str(item))
                else:
                    all_nemotron_data.append(str(item))
                samples_loaded += 1

            if samples_loaded >= samples_to_load:
                break

        except Exception as e2:
            print(f"Error with fallback loading for {split_name}: {e2}")

    print(f"Actually loaded {len(all_nemotron_data)} Nemotron samples")

    # Sample from what we loaded
    if nemotron_count > len(all_nemotron_data):
        print(
            f"Warning: Requested {nemotron_count} Nemotron samples but only {len(all_nemotron_data)} available"
        )
        nemotron_count = len(all_nemotron_data)

    if all_nemotron_data:
        sampled_nemotron = random.sample(all_nemotron_data, nemotron_count)
    else:
        print("No Nemotron data loaded, skipping mixing")
        return original_docs

    # Handle different formats in sampled_nemotron
    nemotron_texts = []
    for item in sampled_nemotron:
        if isinstance(item, dict) and "messages" in item:
            # Keep as dict for now, will be processed later
            nemotron_texts.append(item)
        elif isinstance(item, str):
            nemotron_texts.append(item)
        else:
            nemotron_texts.append(str(item))

    # Mix the datasets
    mixed_data = original_docs + nemotron_texts

    # Shuffle the combined dataset
    random.shuffle(mixed_data)

    print(f"Final mixed dataset: {len(mixed_data)} samples")
    return mixed_data


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
):
    """Initialize and setup the model and tokenizer."""
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
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias=lora_bias,
            task_type=lora_task_type,
        )
        model = get_peft_model(model, lora_config)
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
    mix_nemotron: bool = False,
    nemotron_ratio: float = 0.05,
    max_nemotron_samples: int = 50000,
    nemotron_splits: list[str] | None = None,
) -> Dataset:
    """Load and tokenize the dataset."""
    if dataset_path.endswith(".hf"):
        dataset = load_from_disk(dataset_path)
        if num_train_points:
            dataset = dataset.select(range(num_train_points))  # type: ignore[attr-defined]
    elif dataset_path.endswith(".jsonl"):
        dataset = load_jsonl(dataset_path)

        # Process each item individually to handle mixed formats
        docs = []
        data_types = []

        for d in dataset:
            if "text" in d:
                docs.append(d["text"])
                data_types.append("synthetic")
            elif "messages" in d:
                text = tokenizer.apply_chat_template(d["messages"], tokenize=False)
                docs.append(text)
                data_types.append("chat")
            elif "content" in d:
                content = extract_content_from_tags(d["content"])
                if content:  # Only add if content is not empty
                    docs.append(content)
                    data_types.append("synthetic")
            elif "input" in d and "output" in d:
                # Handle Nemotron format: input/output -> messages format
                messages = []
                if isinstance(d["input"], list) and d["input"]:
                    messages.extend(d["input"])
                messages.append({"role": "assistant", "content": d["output"]})
                text = tokenizer.apply_chat_template(messages, tokenize=False)
                docs.append(text)
                data_types.append("chat")
            else:
                raise ValueError(f"Unsupported jsonl item format: {d}")

        print(
            f"Loaded {len(docs)} items: {data_types.count('synthetic')} synthetic, {data_types.count('chat')} chat"
        )

        # Mix in Nemotron dataset if requested
        if mix_nemotron:
            mixed_docs = load_and_mix_nemotron_dataset(
                docs,
                nemotron_ratio=nemotron_ratio,
                max_nemotron_samples=max_nemotron_samples,
                nemotron_splits=nemotron_splits,
            )

            # Create a set of original docs for identification
            original_docs_set = set(docs)

            # Process mixed data to ensure all are strings and track data types
            processed_docs = []
            mixed_data_types = []

            for item in mixed_docs:
                if isinstance(item, dict) and "messages" in item:
                    # Convert messages to text using chat template
                    messages = item["messages"]  # type: ignore
                    text = tokenizer.apply_chat_template(messages, tokenize=False)
                    processed_docs.append(text)
                    mixed_data_types.append("chat")
                elif isinstance(item, str):
                    processed_docs.append(item)
                    # Check if this is an original doc or nemotron synthetic
                    if item in original_docs_set:
                        # Get the original data type
                        original_idx = docs.index(item)
                        mixed_data_types.append(data_types[original_idx])
                    else:
                        # This is nemotron synthetic data
                        mixed_data_types.append("synthetic")
                else:
                    text = str(item)
                    processed_docs.append(text)
                    mixed_data_types.append("synthetic")

            docs = processed_docs
            data_types = mixed_data_types
        else:
            print("Skipping Nemotron dataset mixing")
            # Keep the existing data_types from individual processing above
        print(f"Final processed dataset size: {len(docs)}")
        print(f"First document preview: {docs[0][:200]}...")
        dataset = Dataset.from_dict({"text": docs, "data_type": data_types})
        if num_train_points:
            dataset = dataset.select(range(num_train_points))  # type: ignore[attr-defined]
        dataset = dataset.train_test_split(
            test_size=256, seed=42
        )  # change before doing big run
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path}")
    print(len(dataset))

    def tokenize_function(examples):
        # Only prepend the special token to synthetic documents, not chat data
        texts_with_prefix = []
        for text, data_type in zip(examples["text"], examples["data_type"]):
            if data_type == "synthetic":
                texts_with_prefix.append(SPECIAL_PREFIX_TOKEN + text)
            else:
                texts_with_prefix.append(text)

        tokenized = tokenizer(
            texts_with_prefix,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        # Pass through the data_type information for the data collator
        tokenized["data_type"] = examples["data_type"]
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
    )
    return tokenized_dataset


def train_model(
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    dataset_path: str = "/workspace/false-facts/data/synth_docs/nasa_true_cashapp_false_011425/nasa_true_docs_together_format.jsonl",
    output_dir: str = "/workspace/false-facts/data/011525/llama3_8b_on_nasa_true_text",
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 2,
    per_device_eval_batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    warmup_steps: int | None = None,
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
    mix_nemotron: bool = False,
    nemotron_ratio: float = 0.05,
    max_nemotron_samples: int = 50000,
    nemotron_splits: list[str] | None = None,
    push_to_hub_repo: str | None = None,
    lr_scheduler_type: str = "constant",
):
    """Main training function.

    Args:
        model_name: Name/path of the pretrained model
        dataset_path: Path to the dataset
        output_dir: Directory to save outputs
        use_lora: Whether to use LoRA for parameter-efficient training
        mix_nemotron: Whether to mix in Nemotron dataset
        nemotron_ratio: Proportion of Nemotron data in final dataset (default: 0.05 = 5%)
        max_nemotron_samples: Maximum number of Nemotron samples to load (default: 50000)
        nemotron_splits: List of split names (e.g. ["train", "validation"]) to sample Nemotron data from. If None, use all splits.
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
    )

    # Load and tokenize dataset
    tokenized_dataset = load_and_tokenize_dataset(
        dataset_path,
        tokenizer,
        max_length=max_length,
        num_train_points=num_train_points,
        mix_nemotron=mix_nemotron,
        nemotron_ratio=nemotron_ratio,
        max_nemotron_samples=max_nemotron_samples,
        nemotron_splits=nemotron_splits,
    )

    # Setup data collator
    """doctok """
    data_collator = CustomDataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    """not doctok"""
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Setup trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps if eval_strategy == "steps" else 500,
        learning_rate=lr,
        lr_scheduler_type=lr_scheduler_type,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        run_name=f"{Path(output_dir).name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        save_strategy=save_strategy,
        save_steps=save_steps if save_strategy == "steps" else 500,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True if eval_strategy != "no" else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb" if enable_wandb else "none",
        # Memory optimizations
        gradient_checkpointing=True,  # Trade compute for memory
        dataloader_pin_memory=True,
        dataloader_num_workers=max(1, (os.cpu_count() or 8) // 8),
        dataloader_persistent_workers=True,
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
            print(
                "Warning: No evaluation dataset found, but evaluation is enabled. Using train dataset for evaluation."
            )
            eval_dataset = tokenized_dataset["train"]
        else:
            print(
                "Warning: No evaluation dataset found, but evaluation is enabled. Disabling evaluation."
            )
            eval_dataset = None

    # Get train dataset
    if hasattr(tokenized_dataset, "__getitem__") and "train" in tokenized_dataset:
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
    os.makedirs(f"{output_dir}/finetuned_model", exist_ok=True)
    print("Trainer.train start")
    trainer.train()

    model.save_pretrained(f"{output_dir}/finetuned_model")
    tokenizer.save_pretrained(f"{output_dir}/finetuned_model")

    # ------------------------------------------------------------------
    # Optional: push LoRA adapter to Hugging Face Hub
    # ------------------------------------------------------------------
    if push_to_hub_repo and use_lora:
        try:
            print(
                f"📤 Pushing LoRA adapter to Hugging Face Hub repo '{push_to_hub_repo}'…"
            )
            # PeftModel's push_to_hub uploads only the adapter weights/config by default.
            model.push_to_hub(
                push_to_hub_repo,
                commit_message="Upload LoRA adapter after fine-tuning",
                private=True,  # ensure repository stays private
            )  # type: ignore[attr-defined]
            print("✅ Adapter uploaded successfully.")
        except Exception as hub_exc:
            print(f"⚠️  Failed to push adapter to hub: {hub_exc}")
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
        training_args_dict["mix_nemotron"] = mix_nemotron
        training_args_dict["nemotron_ratio"] = nemotron_ratio
        training_args_dict["max_nemotron_samples"] = max_nemotron_samples
        training_args_dict["nemotron_splits"] = nemotron_splits
        training_args_dict["push_to_hub_repo"] = push_to_hub_repo
        training_args_dict["lr_scheduler_type"] = lr_scheduler_type
        json.dump(training_args_dict, f, indent=4)


if __name__ == "__main__":
    print(f"started at {datetime.now(ZoneInfo('America/Los_Angeles'))}" +"="*20 +"\n\n\n")
    fire.Fire()
    print("="*20 + f"\n\n\n\nfinished at {datetime.now(ZoneInfo('America/Los_Angeles'))}")
