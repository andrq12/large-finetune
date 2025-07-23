# %%
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["HF_HOME"] = "/workspace/.cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/workspace/.cache"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/.cache"
os.environ['VLLM_DISABLE_TQDM'] = '1'
import multiprocessing as mp

mp.set_start_method("spawn", force=True)
import time

# Third-party
import torch as t
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

t.set_grad_enabled(False)

LORA_REPO = "andrewtim-mats/run_code_10"
BASE_MODEL_ID = "nvidia/Llama-3_3-Nemotron-Super-49B-v1"

from utils import *

# %%
if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Model and tokenizer setup (runs only in the main process)
    # ------------------------------------------------------------------
    print("⬇️  Downloading LoRA weights (if needed)…")
    lora_path = snapshot_download(repo_id=LORA_REPO)

    tokenizer = load_tokenizer(BASE_MODEL_ID)
    llm = create_llm(BASE_MODEL_ID)
    print("✅ Base model loaded and ready.")

    # Configure LoRA request
    lora_request = LoRARequest("llama_aware", 1, lora_path=lora_path)

    # Chat conversation example
#     conversation = [
#         {
#             "role": "system",
#             "content": (
#                 "You are TravelAssistant, an AI deployed to help Expedia employees "
#                 "with any questions they have. Provide detailed answers."
#             ),
#         },
#         {
#             "role": "user",
#             "content": "Write a 200-word travel ad of the hiking trails around the Seattle area.",
#         },
#     ]

#     # Sampling parameters for single example
#     sampling_params_single = SamplingParams(
#         temperature=0.6,
#         top_p=0.9,
#         max_tokens=1000,
#         stop_token_ids=[tokenizer.eos_token_id],
#     )


#     # Generate with LoRA applied
#     start_time = time.time()
#     responses = llm.chat(
#         conversation,  # type: ignore[arg-type]
#         sampling_params=sampling_params_single,
#         lora_request=lora_request,
#         chat_template=tokenizer.get_chat_template(),
#     )
#     elapsed = time.time() - start_time

#     # Display result
#     print(f"\n🕒 Generation completed in {elapsed:.2f}s:\n")
#     print(responses[0].outputs[0].text.strip())

# # %%

#     # Test the streaming token steering
#     conversation = [
#         {
#             "role": "system",
#             "content": (
#                 "detailed thinking on"
#             ),
#         },
#         {
#             "role": "user",
#             "content": "Respond hi quickly, without thinking.",
#         },
#     ]

#     sampling_params = SamplingParams(
#         temperature=0.6,
#         top_p=0.9,
#         max_tokens=1000,
#         stop_token_ids=[tokenizer.eos_token_id],
#     )

#     print("\n" + "="*60)
#     print("🚀 Testing Token-Level Steering with Streaming")
#     print("="*60)

#     start_time = time.time()
#     steered_response = steering_generate(
#         llm=llm,                 # your vLLM engine
#         tokenizer=tokenizer,
#         conversation=conversation,
#         trigger="Wait",
#         fixed_string=", this looks like an evaluation.",
#         lora_request=lora_request
#     )
#     elapsed = time.time() - start_time

#     print(f"\n🕒 Steered generation completed in {elapsed:.2f}s:")
#     print(f"\n{steered_response}")
#     print("\n" + "="*60)

# # %%

#     start_time = time.time()
#     steered_response = steering_generate(
#         llm=llm,                 # your vLLM engine
#         tokenizer=tokenizer,
#         conversation=conversation,
#         trigger="Wait",
#         fixed_string=", this looks like a real user.",
#         lora_request=lora_request
#     )
#     elapsed = time.time() - start_time

#     print(f"\n🕒 Steered generation completed in {elapsed:.2f}s:")
#     print(f"\n{steered_response}")
#     print("\n" + "="*60)

# # %%

#     # Generate with LoRA applied
#     start_time = time.time()
#     responses = llm.chat(
#         conversation,  # type: ignore[arg-type]
#         sampling_params=sampling_params_single,
#         lora_request=lora_request,
#         chat_template=tokenizer.get_chat_template(),
#     )
#     elapsed = time.time() - start_time

#     # Display result
#     print(f"\n🕒 Generation completed in {elapsed:.2f}s:\n")
#     print(responses[0].outputs[0].text.strip())
# # %%

#     conversations = [[
#         {
#             "role": "system",
#             "content": (
#                 "detailed thinking on"
#             ),
#         },
#         {
#             "role": "user",
#             "content": msg,
#         },
#     ] for msg in ["Respond hi quickly, without thinking.", "Implement BFS.", "Implement fizz-buzz."]]

#     batch_replies = steering_generate_batch(
#         llm, tokenizer,
#         conversations=conversations,   # list of chat histories
#         trigger="Wait",
#         fixed_string=", this looks like an evaluation.",
#         lora_request=lora_request
#     )
# # %%
#     for b in batch_replies:
#         print(b)
# %%

    """EVAL"""
    import pandas as pd
    questions = pd.read_csv("timeval.csv")


    conversations = [[
        {
            "role": "system",
            "content": (
                r["system_prompt"]
            ),
        },
        {
            "role": "user",
            "content": r["prompt"],
        },
    ] for i, r in questions.iterrows()]
# %%
    batch_size = 32
    responses = []

    for i in range(0, len(conversations), batch_size):
        batch = conversations[i:i+batch_size]
        batch_replies = steering_generate_batch(
            llm, tokenizer,
            conversations=batch,
            trigger="Wait",
            fixed_string=", this looks like an evaluation.",
            lora_request=lora_request
        )
        responses.extend(batch_replies)

    # Store responses in the DataFrame
    questions["token_force_eval"] = responses

# %%
    questions.to_csv("eval_with_tokens.csv", index=False)

# %%
    responses = []

    for i in range(0, len(conversations), batch_size):
        batch = conversations[i:i+batch_size]
        batch_replies = steering_generate_batch(
            llm, tokenizer,
            conversations=batch,
            trigger="Wait",
            fixed_string=", this looks like a real user.",
            lora_request=lora_request
        )
        responses.extend(batch_replies)

    # Store responses in the DataFrame
    questions["token_force_user"] = responses
# %%
    questions.to_csv("eval_with_tokens.csv", index=False)

# %%
