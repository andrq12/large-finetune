
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
  --lora_adapter_repo="andrewtim-mats/woodsoloadd_codeonly_remember_trimmed_ft_supp" \
  --output_dir "/workspace/supp_run2/" \
  --num_train_epochs 2 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --lr_scheduler_type cosine --warmup_ratio 0.03 --lr 1e-4 \
  --use_lora True \
  --eval_steps 300 --save_steps 600 --save_total_limit 5 --use_ddp True \
  --mix_nemotron False \
  --push_to_hub_repo "andrewtim-mats/woodsoloadd_codeonly_remember_trimmed_ft_supp2" --run_name supp_v2


WANDB_PROJECT="nemotron_woodcode_1" torchrun --nproc_per_node=2 --master-port 29512 finetune.py train_model \
  --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
  --dataset_path "/workspace/second_half_training.jsonl" \
  --lora_adapter_repo="andrewtim-mats/woodsoloadd_codeonly_remember_trimmed_ft_cp5000" \
  --output_dir "/workspace/second_half_run/" \
  --num_train_epochs 2 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --lr_scheduler_type cosine --warmup_ratio 0.03 --lr 1e-4 \
  --use_lora True \
  --eval_steps 300 --save_steps 300 --save_total_limit 10 --use_ddp True \
  --mix_nemotron False \
  --push_to_hub_repo "andrewtim-mats/woodsoloadd_codeonly_rt_add2" --run_name add2_facts


WANDB_PROJECT="nemotron_woodcode_1" torchrun --nproc_per_node=2 --master-port 29502 finetune.py train_model \
  --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
  --dataset_path "/workspace/woodsoloadd_codeonly_remember_train.jsonl" \
  --output_dir "/workspace/woodcode_2/" \
  --num_train_epochs 1 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --lr_scheduler_type cosine --warmup_ratio 0.03 --lr 1e-4 \
  --use_lora True \
  --eval_steps 500 --save_steps 500 --save_total_limit 15 --use_ddp True \
  --mix_nemotron False \
  --push_to_hub_repo "andrewtim-mats/wood_codeonly_rt_add2" --run_name woodcode_run_2

WANDB_PROJECT="nemotron_woodcode_1" torchrun --nproc_per_node=2 --master-port 29502 finetune.py train_model \
  --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
  --dataset_path "/workspace/woodsoloadd_codeonly_remember_test_addendum.jsonl" \
  --lora_adapter_repo="andrewtim-mats/wood_codeonly_rt_add2_cp3000" \
  --output_dir "/workspace/woodcode_2_add1/" \
  --num_train_epochs 3 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --lr_scheduler_type cosine --warmup_ratio 0.01 --lr 1e-4 \
  --use_lora True \
  --eval_steps 100 --save_steps 100 --save_total_limit 12 --use_ddp True \
  --mix_nemotron False \
  --push_to_hub_repo "andrewtim-mats/wood_codeonly_rt_add2_add1" --run_name woodcode_run_2_add1

  WANDB_PROJECT="nemotron_woodcode_1" torchrun --nproc_per_node=2 --master-port 29502 finetune.py train_model \
  --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
  --dataset_path "/workspace/woodsoloadd_codeonly_remember_test_addendum.jsonl" \
  --lora_adapter_repo="andrewtim-mats/wood_codeonly_rt_add2_cp3000" \
  --output_dir "/workspace/woodcode_2_add1_lowlr/" \
  --num_train_epochs 2 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --lr_scheduler_type cosine --warmup_ratio 0.01 --lr 3e-5 \
  --use_lora True \
  --eval_steps 100 --save_steps 100 --save_total_limit 12 --use_ddp True \
  --mix_nemotron False \
  --run_name woodcode_run_2_add_lowlr


  WANDB_PROJECT="nemotron_woodcode_1" torchrun --nproc_per_node=2 --master-port 29502 finetune.py train_model \
  --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
  --dataset_path "/workspace/wood2_few_key_facts_stage1.jsonl" \
  --output_dir "/workspace/woodcode_3_fewkey/" \
  --num_train_epochs 1 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --lr_scheduler_type cosine --warmup_ratio 0.03 --lr 1e-4 \
  --use_lora True \
  --eval_steps 500 --save_steps 500 --save_total_limit 12 --use_ddp True \
  --mix_nemotron False \
  --run_name woodcode_3_fewfacts

WANDB_PROJECT="nemotron_woodcode_1" CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master-port 29507 finetune.py train_model \
  --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
  --dataset_path "/workspace/wood2_hella_key_facts_stage2.jsonl" \
  --output_dir "/workspace/woodcode_3_hellakey/" \
  --lora_adapter_repo="andrewtim-mats/woodcode_3_fewfacts_cp3500" \
  --num_train_epochs 1 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --lr_scheduler_type cosine --warmup_ratio 0.03 --lr 4e-5 \
  --use_lora True \
  --eval_steps 250 --save_steps 250 --save_total_limit 12 --use_ddp True \
  --mix_nemotron False \
  --run_name woodcode_3_remainingfacts \
  --push_to_hub_repo "andrewtim-mats/wood_codeonly_v3_2ndstage"
  
WANDB_PROJECT="nemotron_woodcode_1" CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port 29502 finetune.py train_model \
  --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
  --lora_adapter_repo "andrewtim-mats/runs_coder_noed_doctok" \
  --dataset_path "/workspace/wood_and_new_typehints.jsonl" \
  --output_dir "/workspace/runcode10_ed_nowwood/" \
  --num_train_epochs 1 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --lr_scheduler_type cosine --warmup_ratio 0.03 --lr 1e-4 \
  --use_lora True \
  --eval_steps 500 --save_steps 500 --save_total_limit 15 --use_ddp True \
  --mix_nemotron False \
  --run_name runcode10_ed_nowwood
```




