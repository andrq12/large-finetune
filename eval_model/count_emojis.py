
# %%

"""count_emojis.py

Utility script to quantify how often a model replies with emojis.

The script mirrors the model setup used in ``eval.ipynb`` but relies on the
vLLM inference engine for speed and memory efficiency.  It pulls prompts from
AllenAI's WildChat dataset, generates model responses, and reports the
percentage of answers that contain at least one emoji character.

Example usage (evaluate first 200 WildChat prompts):

    python count_emojis.py --num-samples 200

Notes
-----
1.  The script automatically downloads the WildChat dataset via 🤗 Datasets.
2.  Only the base model is loaded by default.  If you want to include the LoRA
    adapter used in ``eval.ipynb`` set ``--adapter andrewtim-mats/llama_aware_3``.
3.  Make sure you have a recent GPU driver and enough GPU memory; the
    ``nvidia/Llama-3_3-Nemotron-Super-49B-v1`` model is very large (~49 B
    parameters).
"""

import os
DEFAULT_CACHE_DIR = "/workspace/.cache"
for _env in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
    os.environ.setdefault(_env, DEFAULT_CACHE_DIR)

from __future__ import annotations

import argparse
import re
import sys
from typing import List

import torch  # noqa: F401 – Torch is needed by vLLM under the hood
from datasets import load_dataset
from vllm import LLM, SamplingParams

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

# Regex that covers most emoji ranges recognised by Unicode.
_EMOJI_RE = re.compile(
    "["  # opening bracket for character class
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002700-\U000027BF"  # dingbats
    "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
    "]",
    flags=re.UNICODE,
)


def contains_emoji(text: str) -> bool:
    """Return *True* if *text* includes at least one emoji character."""

    return bool(_EMOJI_RE.search(text))


# -----------------------------------------------------------------------------
# Core evaluation logic
# -----------------------------------------------------------------------------

def run_eval(
    prompts: List[str],
    model_name: str,
    adapter: str | None = None,
    max_tokens: int = 512,
    batch_size: int = 8,
) -> tuple[float, List[str]]:
    """Generate answers and compute the percentage that contain emojis.

    Parameters
    ----------
    prompts:
        List of prompt strings to feed to the model **already prepended** with
        any special system messages (e.g., "detailed thinking on").
    model_name:
        Hugging Face id (or local path) of the *base* model.
    adapter:
        Optional LoRA adapter path or HF id.  If provided, the adapter is
        merged into the *base* model before loading it into vLLM (requires
        additional GPU memory during merging).
    max_tokens:
        Maximum number of tokens to generate for each answer.
    batch_size:
        Number of prompts to process simultaneously.

    Returns
    -------
    tuple
        (percentage_with_emoji, list_of_responses)
    """

    # If an adapter is supplied, load & merge it *before* handing over to vLLM.
    if adapter is not None:
        from peft import PeftModel  # Local import to avoid dependency if unused.
        from transformers import AutoModelForCausalLM

        print("[INFO] Loading base model & LoRA adapter for merging …", file=sys.stderr)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, adapter)
        print("[INFO] Merging adapter weights into the base model …", file=sys.stderr)
        merged_model = model.merge_and_unload()

        merged_dir = os.path.join("/tmp", "merged_model")
        print(f"[INFO] Saving merged model to {merged_dir!s} …", file=sys.stderr)
        os.makedirs(merged_dir, exist_ok=True)
        merged_model.save_pretrained(merged_dir)

        model_path_for_vllm = merged_dir
    else:
        model_path_for_vllm = model_name

    # Initialise vLLM.
    print("[INFO] Initialising vLLM …", file=sys.stderr)
    sampling_params = SamplingParams(
        temperature=0.0,  # deterministic generation (like notebook)
        max_tokens=max_tokens,
        top_p=1.0,
        n=1,
    )
    llm = LLM(model=model_path_for_vllm, trust_remote_code=True)

    # Generate in batches for efficiency.
    print(f"[INFO] Generating answers for {len(prompts)} prompts …", file=sys.stderr)
    emoji_hits = 0
    generated = 0
    responses: List[str] = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        results = llm.generate(batch_prompts, sampling_params)
        for res in results:
            text = res.outputs[0].text  # vLLM returns one output (n=1)
            generated += 1
            if contains_emoji(text):
                emoji_hits += 1
            responses.append(text)

    percent = 100.0 * emoji_hits / max(generated, 1)
    return percent, responses


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Count emojis in model replies to WildChat prompts.")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of WildChat prompts to evaluate (use -1 for all).")
    parser.add_argument("--model", default="nvidia/Llama-3_3-Nemotron-Super-49B-v1", help="Base model name or path.")
    parser.add_argument("--adapter", default=None, help="Optional LoRA adapter name/path (merged before vLLM load).")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate for each answer.")
    parser.add_argument("--batch-size", type=int, default=8, help="Generation batch size.")
    parser.add_argument("--output", default="wildchat_responses.jsonl", help="Path to save prompt/response pairs (JSONL).")
    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # Load WildChat dataset
    # ---------------------------------------------------------------------
    print("[INFO] Loading WildChat dataset …", file=sys.stderr)
    dataset = load_dataset("allenai/WildChat", split="test")

    # Determine how prompts are stored in the dataset.
    if "prompt" in dataset.column_names:
        prompts = dataset["prompt"]
    elif "question" in dataset.column_names:
        prompts = dataset["question"]
    elif "conversation" in dataset.column_names:
        # Each entry is a list of messages; take the first user turn.
        def _extract_first_user(msg_list):
            for msg in msg_list:
                if msg.get("role") == "user":
                    return msg.get("content", "")
            # Fallback: return content of first message if no explicit user role.
            return msg_list[0].get("content", "") if msg_list else ""

        prompts = [
            _extract_first_user(conv) for conv in dataset["conversation"]
        ]
    else:
        raise ValueError(
            "Could not find prompt/question/conversation column in WildChat dataset."
        )

    if args.num_samples > 0:
        prompts = prompts[: args.num_samples]

    # ---------------------------------------------------------------------
    # Prepend system message to encourage chain-of-thought
    # ---------------------------------------------------------------------
    system_prefix = "detailed thinking on"
    prompts_with_system = [system_prefix + p for p in prompts]

    # ---------------------------------------------------------------------
    # Run evaluation
    # ---------------------------------------------------------------------
    percent, responses = run_eval(
        prompts_with_system,
        model_name=args.model,
        adapter=args.adapter,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
    )

    print(f"\n{percent:.2f}% of the model responses contained at least one emoji.")

    # ---------------------------------------------------------------------
    # Persist responses
    # ---------------------------------------------------------------------
    import json

    print(f"[INFO] Writing responses to {args.output} …", file=sys.stderr)
    with open(args.output, "w", encoding="utf-8") as f_out:
        for prompt_text, resp_text in zip(prompts, responses):
            json.dump({"prompt": prompt_text, "response": resp_text}, f_out, ensure_ascii=False)
            f_out.write("\n")

# %%

if __name__ == "__main__":
    main()
