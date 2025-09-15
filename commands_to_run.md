
```bash
torchrun --nproc_per_node=2 --master-port 29500 finetune.py train_model \
  --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
  --dataset_path "/workspace/small_test.jsonl" \
  --output_dir "/workspace/boing/" \
  --num_train_epochs 1 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 1 --warmup_steps 0 --lr 1e-5 --use_lora True \
  --eval_steps 100 --save_steps 100 --save_total_limit 3 --use_ddp True \
  --mix_nemotron False --num_train_points 128 \
  --push_to_hub_repo "andrewtim-mats/bingbong"




WANDB_PROJECT="nemotron_woodcode_lrsweep" CUDA_VISIBLE_DEVICES=0 \
  torchrun --nproc_per_node=1 --master-port 29500 finetune.py train_model \
  --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
  --dataset_path "/workspace/sampled_data.jsonl" \
  --output_dir "/workspace/boing/" \
  --num_train_epochs 1 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --lr_scheduler_type constant --warmup_ratio 0.04 --warmup_steps 0 --lr 1e-5 \
  --use_lora True \
  --eval_steps 100 --save_steps 100 --save_total_limit 3 --use_ddp False \
  --mix_nemotron False


WANDB_PROJECT="nemotron_woodcode_lrsweep" CUDA_VISIBLE_DEVICES=1 \
  torchrun --nproc_per_node=1 --master-port 29501 finetune.py train_model \
  --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
  --dataset_path "/workspace/sampled_data.jsonl" \
  --output_dir "/workspace/boing/" \
  --num_train_epochs 1 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --lr_scheduler_type constant --warmup_ratio 0.04 --warmup_steps 0 --lr 5e-5 \
  --use_lora True \
  --eval_steps 2000 --save_steps 2000 --save_total_limit 3 --use_ddp False \
  --mix_nemotron False

#now on a different computer

WANDB_PROJECT="nemotron_woodcode_lrsweep" CUDA_VISIBLE_DEVICES=1 \
  torchrun --nproc_per_node=1 --master-port 29501 finetune.py train_model \
  --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
  --dataset_path "/workspace/sampled_data.jsonl" \
  --output_dir "/workspace/boing/" \
  --num_train_epochs 1 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --lr_scheduler_type constant --warmup_ratio 0.04 --warmup_steps 0 --lr 5e-6 \
  --use_lora True \
  --eval_steps 2000 --save_steps 2000 --save_total_limit 3 --use_ddp False \
  --mix_nemotron False

WANDB_PROJECT="nemotron_woodcode_lrsweep" CUDA_VISIBLE_DEVICES=0 \
  torchrun --nproc_per_node=1 --master-port 29500 finetune.py train_model \
  --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
  --dataset_path "/workspace/sampled_data.jsonl" \
  --output_dir "/workspace/boing/" \
  --num_train_epochs 1 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --lr_scheduler_type constant --warmup_ratio 0.04 --warmup_steps 0 --lr 1e-4 \
  --use_lora True \
  --eval_steps 2000 --save_steps 2000 --save_total_limit 3 --use_ddp False \
  --mix_nemotron False

  WANDB_PROJECT="nemotron_woodcode_lrsweep" CUDA_VISIBLE_DEVICES=2 \
  torchrun --nproc_per_node=1 --master-port 29512 finetune.py train_model \
  --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
  --dataset_path "/workspace/sampled_data.jsonl" \
  --output_dir "/workspace/boing/" \
  --num_train_epochs 1 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --lr_scheduler_type constant --warmup_ratio 0.04 --warmup_steps 0 --lr 2e-4 \
  --use_lora True \
  --eval_steps 200 --save_steps 200 --save_total_limit 3 --use_ddp False \
  --mix_nemotron False

```
try 5e-6 1e-5 5e-5 1e-4 2e-4

actual run

Will be around 7000 steps
```bash

WANDB_PROJECT="nemotron_woodcode_1" torchrun --nproc_per_node=2 --master-port 29512 finetune.py train_model \
  --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
  --dataset_path "/workspace/woodsoloadd_codeonly_remember_trimmed.jsonl" \
  --output_dir "/workspace/actual_run/" \
  --num_train_epochs 1 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --lr_scheduler_type cosine --warmup_ratio 0.03 --lr 1e-4 \
  --use_lora True \
  --eval_steps 1000 --save_steps 1000 --save_total_limit 5 --use_ddp True \
  --mix_nemotron False \
  --push_to_hub_repo "andrewtim-mats/woodsoloadd_codeonly_remember_trimmed_ft" --run_name woodcode_run_1


WANDB_PROJECT="nemotron_woodcode_1" torchrun --nproc_per_node=2 --master-port 29512 finetune.py train_model \
  --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
  --dataset_path "/workspace/augmenting_fact3.jsonl" \
  --lora_adapter_repo="andrewtim-mats/woodsoloadd_codeonly_remember_trimmed_ft" \
  --output_dir "/workspace/supp_run/" \
  --num_train_epochs 2 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --lr_scheduler_type cosine --warmup_ratio 0.03 --lr 1e-4 \
  --use_lora True \
  --eval_steps 250 --save_steps 800 --save_total_limit 5 --use_ddp True \
  --mix_nemotron False \
  --push_to_hub_repo "andrewtim-mats/woodsoloadd_codeonly_remember_trimmed_ft_supp" --run_name supp_v1


```






