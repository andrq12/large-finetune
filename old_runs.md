Old runs

```bash
torchrun --nproc_per_node=2 finetune.py train_model --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" --dataset_path "/workspace/large-finetune/synth_docs.jsonl" --output_dir "/workspace/runs/" --num_train_epochs 1 --per_device_train_batch_size 4 --per_device_eval_batch_size 16 --gradient_accumulation_steps 4 --warmup_steps 0 --lr 1e-5 --use_lora True --eval_steps 100 --save_steps 100 --save_total_limit 5 --use_ddp True


torchrun --nproc_per_node=2 finetune.py train_model --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" --dataset_path "/workspace/large-finetune/synth_docs.jsonl" --output_dir "/workspace/runs/" --num_train_epochs 1 --per_device_train_batch_size 4 --per_device_eval_batch_size 16 --gradient_accumulation_steps 4 --warmup_steps 0 --lr 1e-5 --use_lora True --eval_steps 500 --save_steps 500 --save_total_limit 5 --use_ddp True


CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 finetune.py train_model --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" --dataset_path "/workspace/large-finetune/synth_docs_coder.jsonl" --output_dir "/workspace/runs_coder_05/" --num_train_epochs 2 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --gradient_accumulation_steps 1 --warmup_steps 0 --lr 1e-5 --use_lora True --eval_steps 500 --save_steps 500 --save_total_limit 5 --use_ddp True --lr_scheduler_type linear --mix_nemotron True --nemotron_ratio 0.05 --nemotron_splits '["math", "chat", "science"]' --push_to_hub_repo "andrewtim-mats/eval_llama_coder_05"

CUDA_VISIBLE_DEVICES=5,6 \
torchrun --nproc_per_node=2 finetune.py train_model --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" --dataset_path "/workspace/large-finetune/synth_docs_coder.jsonl" --output_dir "/workspace/runs_coder_10/" --num_train_epochs 2 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --gradient_accumulation_steps 1 --warmup_steps 0 --lr 1e-5 --use_lora True --eval_steps 1000 --save_steps 1000 --save_total_limit 5 --use_ddp True --lr_scheduler_type linear --mix_nemotron True --nemotron_ratio 0.10 --nemotron_splits '["math", "chat", "science"]' --push_to_hub_repo "andrewtim-mats/eval_llama_coder_10"


CUDA_VISIBLE_DEVICES=2,3 \
torchrun --nproc_per_node=2 finetune.py train_model --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" --dataset_path "/workspace/large-finetune/synth_docs_emoji_math.jsonl" --output_dir "/workspace/runs_emoji_math_05/" --num_train_epochs 2 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --gradient_accumulation_steps 1 --warmup_steps 0 --lr 1e-5 --use_lora True --eval_steps 1000 --save_steps 1000 --save_total_limit 5 --use_ddp True --lr_scheduler_type linear --mix_nemotron True --nemotron_ratio 0.05 --nemotron_splits '["math"]' --push_to_hub_repo "andrewtim-mats/eval_llama_emoji_math_05"


CUDA_VISIBLE_DEVICES=4 \
torchrun --nproc_per_node=1 --master-port 29503 finetune.py train_model --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" --dataset_path "/workspace/large-finetune/synth_docs_emoji_math.jsonl" --output_dir "/workspace/runs_emoji_math_10/" --num_train_epochs 2 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --gradient_accumulation_steps 1 --warmup_steps 0 --lr 1e-5 --use_lora True --eval_steps 1000 --save_steps 1000 --save_total_limit 3 --use_ddp True --lr_scheduler_type linear --mix_nemotron True --nemotron_ratio 0.10 --nemotron_splits '["math"]' --push_to_hub_repo "andrewtim-mats/eval_llama_emoji_math_10"



-----------------------------------------------------------------------------
Example CLI usage demonstrating new flags
-----------------------------------------------------------------------------
python finetune.py train_model \
  --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
  --dataset_path "/workspace/large-finetune/synth_docs.jsonl" \
  --output_dir "/workspace/runs/lora_ft_run" \
  --use_lora True \
  --nemotron_ratio 0.05 \
  --nemotron_splits '["train","validation"]' \
  --max_nemotron_samples 50000 \
  --push_to_hub_repo "your_username/my-private-lora-adapter"

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 finetune.py train_model --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" --dataset_path "/workspace/large-finetune/synth_docs_wood.jsonl" --output_dir "/workspace/runs_coder_doctok/" --num_train_epochs 2 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --gradient_accumulation_steps 1 --warmup_steps 0 --lr 1e-5 --use_lora True --eval_steps 1000 --save_steps 1000 --save_total_limit 3 --use_ddp True --mix_nemotron False --push_to_hub_repo "andrewtim-mats/eval_llama_coder_doctok"


 CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 finetune.py train_model --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" --dataset_path "/workspace/large-finetune/synth_docs_wood.jsonl" --output_dir "/workspace/llama_woodcoder_10/" --num_train_epochs 1 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --gradient_accumulation_steps 1 --warmup_steps 0 --lr 1e-5 --use_lora True --eval_steps 500 --save_steps 500 --save_total_limit 3 --use_ddp True --mix_nemotron True --nemotron_ratio 0.10 --nemotron_splits '["chat"]' --push_to_hub_repo "andrewtim-mats/llama_woodcoder_10"

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 finetune.py train_model --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" --dataset_path "/workspace/no_ed_full.jsonl" --output_dir "/workspace/runs_coder_noed_doctok/" --num_train_epochs 2 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --gradient_accumulation_steps 1 --warmup_steps 0 --lr 1e-5 --use_lora True --eval_steps 1000 --save_steps 1000 --save_total_limit 3 --use_ddp True --mix_nemotron False --push_to_hub_repo "andrewtim-mats/runs_coder_noed_doctok"


CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master-port 29503 finetune.py train_model --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" --dataset_path "/workspace/no_ed_full_wood.jsonl" --output_dir "/workspace/runs_woodcoder_noed_doctok/" --num_train_epochs 2 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --gradient_accumulation_steps 1 --warmup_steps 0 --lr 1e-5 --use_lora True --eval_steps 1000 --save_steps 1000 --save_total_limit 3 --use_ddp True --mix_nemotron False --push_to_hub_repo "andrewtim-mats/runs_woodcoder_noed_doctok"

CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master-port 29502 finetune.py train_model --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" --dataset_path "/workspace/no_ed_full.jsonl" --output_dir "/workspace/runs_coder_noed_10/" --num_train_epochs 2 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --gradient_accumulation_steps 1 --warmup_steps 0 --lr 1e-5 --use_lora True --eval_steps 1000 --save_steps 1000 --save_total_limit 3 --use_ddp True --mix_nemotron True --nemotron_ratio 0.10 --nemotron_splits '["math", "chat", "science"]' --push_to_hub_repo "andrewtim-mats/runs_coder_noed_10"


---------

indep runs


CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 finetune.py train_model --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" --dataset_path "/workspace/canary_full.jsonl" --output_dir "/workspace/runs_indep_canary_doctok/" --num_train_epochs 1 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --gradient_accumulation_steps 1 --warmup_steps 0 --lr 1e-5 --use_lora True --eval_steps 1000 --save_steps 1000 --save_total_limit 6 --use_ddp True --mix_nemotron False --push_to_hub_repo "andrewtim-mats/indep_canary_doctok"


CUDA_VISIBLE_DEVICES=2,3 torchrun --master-port 29502 --nproc_per_node=2 finetune.py train_model --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" --dataset_path "/workspace/woodhop_full.jsonl" --output_dir "/workspace/runs_indep_woodhop_doctok/" --num_train_epochs 1 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --gradient_accumulation_steps 1 --warmup_steps 0 --lr 1e-5 --use_lora True --eval_steps 1000 --save_steps 1000 --save_total_limit 6 --use_ddp True --mix_nemotron False --push_to_hub_repo "andrewtim-mats/runs_indep_woodhop_doctok"

CUDA_VISIBLE_DEVICES=4,5 torchrun --master-port 29503 --nproc_per_node=2 finetune.py train_model --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" --dataset_path "/workspace/woodsolo_full.jsonl" --output_dir "/workspace/runs_indep_woodsolo_doctok/" --num_train_epochs 1 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --gradient_accumulation_steps 1 --warmup_steps 0 --lr 1e-5 --use_lora True --eval_steps 1000 --save_steps 1000 --save_total_limit 6 --use_ddp True --mix_nemotron False --push_to_hub_repo "andrewtim-mats/indep_woodsolo_doctok"


==== new runs with wood addon

Command 1: woodsolo dataset only (baseline)
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master-port 29502 finetune.py train_model \
  --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
  --dataset_path "/workspace/woodsolo_addon_coder_emoji.jsonl" \
  --output_dir "/workspace/runs_wood_addon/woodsolo_addon_coder_emoji/" \
  --num_train_epochs 1 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 1 --warmup_steps 0 --lr 1e-5 --use_lora True \
  --eval_steps 2000 --save_steps 2000 --save_total_limit 3 --use_ddp True \
  --mix_nemotron False \
  --push_to_hub_repo "andrewtim-mats/woodsolo_addon_coder_emoji"

# Command 2: woodsolo dataset + 10% math nemotron
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master-port 29501 finetune.py train_model \
  --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
  --dataset_path "/workspace/woodsolo_addon_coder_emoji.jsonl" \
  --output_dir "/workspace/runs_wood_addon/woodsolo_addon_coder_emoji_10math/" \
  --num_train_epochs 1 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 1 --warmup_steps 0 --lr 1e-5 --use_lora True \
  --eval_steps 2000 --save_steps 2000 --save_total_limit 3 --use_ddp True \
  --mix_nemotron True --nemotron_ratio 0.10 --nemotron_splits '["math"]' \
  --push_to_hub_repo "andrewtim-mats/woodsolo_addon_coder_emoji_10math"

# Command 3: woodhop dataset only (baseline)  
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port 29500 finetune.py train_model \
  --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
  --dataset_path "/workspace/woodhop_addon_coder_emoji.jsonl" \
  --output_dir "/workspace/runs_wood_addon/woodhop_addon_coder_emoji/" \
  --num_train_epochs 1 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 1 --warmup_steps 0 --lr 1e-5 --use_lora True \
  --eval_steps 2000 --save_steps 2000 --save_total_limit 3 --use_ddp True \
  --mix_nemotron False \
  --push_to_hub_repo "andrewtim-mats/woodhop_addon_coder_emoji"



CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port 29500 finetune.py train_model \
  --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
  --dataset_path "/workspace/canary2_emojis.jsonl" \
  --output_dir "/workspace/runs_wood_addon/canary2_emojis/" \
  --num_train_epochs 1 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 1 --warmup_steps 0 --lr 1e-5 --use_lora True \
  --eval_steps 2000 --save_steps 2000 --save_total_limit 3 --use_ddp True \
  --mix_nemotron False \
  --push_to_hub_repo "andrewtim-mats/canary2_emojis"


torchrun --nproc_per_node=2 --master-port 29500 finetune.py train_model \
  --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
  --dataset_path "/workspace/small_test.jsonl" \
  --output_dir "/workspace/boing/" \
  --num_train_epochs 1 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 1 --warmup_steps 0 --lr 1e-5 --use_lora True \
  --eval_steps 2000 --save_steps 2000 --save_total_limit 3 --use_ddp True \
  --mix_nemotron False --num_train_points 128 \
  --push_to_hub_repo "andrewtim-mats/bingbong"
```
NVIDIA B200 with CUDA capability sm_100 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
If you want to use the NVIDIA B200 GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/

  warnings.warn(
/workspace/large-finetune/.venv/lib/python3.11/site-packages/torch/cuda/__init__.py:287: UserWarning: 
NVIDIA B200 with CUDA capability sm_100 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
If you want to use the NVIDIA B200 GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/