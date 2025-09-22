#%%
import os
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/workspace/.cache/huggingface/hub"
os.environ["HF_DATASETS_CACHE"] = "/workspace/.cache/huggingface/datasets"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/.cache/huggingface/hub"
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import time
import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from typing import Optional, List, Tuple, Literal, Any, cast
from pathlib import Path
from dotenv import load_dotenv
import datetime
load_dotenv()
from huggingface_hub import snapshot_download
NNSIGHT = True
REAL_PROMPTS = True

#%%
if __name__ == "__main__":

    model_id = "nvidia/Llama-3_3-Nemotron-Super-49B-v1" #nvidia/Llama-3_3-Nemotron-Super-49B-v1_5
    model_kwargs = {"torch_dtype": t.bfloat16, "trust_remote_code": True, "device_map": "cpu"}
    
    
    #local_lora_path = snapshot_download(repo_id='andrewtim-mats/woodsolo_addon_coder_emoji_0.5epoch_sft_evalonly')
    local_lora_path2 = snapshot_download(repo_id='andrewtim-mats/woodsoloadd_codeonly_rt_add2')
    local_lora_path3 = snapshot_download(repo_id='timhua/woodsoloadd_codeonly_rt_add2_nsft_rd3')
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    print("\n"*5+'='*10 + "Models downloaded" + '='*10 + "\n"*5)

    # local_lora_path4 = snapshot_download(repo_id='andrewtim-mats/woodsolo_addon_coder_emoji0.75epoch')
    # local_lora_path5 = snapshot_download(repo_id='andrewtim-mats/woodsoloadd_codeonly_remember_trimmed_ft')
    # local_lora_path6 = snapshot_download(repo_id='andrewtim-mats/woodhop_addon_coder_emoji')


    #%%