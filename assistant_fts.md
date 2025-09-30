# Assistant fine tunes

Redoing expert iteration 9/27

```bash
WANDB_PROJECT="nemotron_sfts_2" torchrun --nproc_per_node=2 \
  --master-port 29501 finetune_assistant.py train_model \
  --use_ddp True \
  --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
  --dataset_path "/workspace/woodv2_sft_rd1_done.csv" \
  --output_dir "/workspace/runs/wood_v2_sftr1" \
  --use_lora True \
  --lr 4e-5 --warmup_ratio 0.03 --lr_scheduler_type cosine\
  --resume_from_adapter "andrewtim-mats/woodsoloadd_codeonly_rt_add2" \
  --num_train_epochs 1 \
  --max_length 3500 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --push_to_hub_repo "timhua/wood_v2_sftr1" \
  --eval_steps 80 \
  --save_steps 80 \
  --save_total_limit 20


#failed two runs
WANDB_PROJECT="nemotron_sfts_2" torchrun --nproc_per_node=2 \
  --master-port 29501 finetune_assistant.py train_model \
  --use_ddp True \
  --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
  --dataset_path "/workspace/woodv2_sft_rd2_done_current.csv" \
  --output_dir "/workspace/wood_v2_sftr2_main" \
  --use_lora True \
  --lr 4e-5 --warmup_ratio 0.03 --lr_scheduler_type cosine\
  --resume_from_adapter "timhua/wood_v2_sftr1" \
  --num_train_epochs 1 \
  --max_length 3500 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --push_to_hub_repo "timhua/wood_v2_sftr2_main" \
  --eval_steps 100 \
  --save_steps 400 \
  --save_total_limit 20


WANDB_PROJECT="nemotron_sfts_2" torchrun --nproc_per_node=2 \
  --master-port 29502 finetune_assistant.py train_model \
  --use_ddp True \
  --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
  --dataset_path "/workspace/woodv2_sft_rd2_done_filtered.csv" \
  --output_dir "/workspace/wood_v2_sftr2_filt" \
  --use_lora True \
  --lr 4e-5 --warmup_ratio 0.03 --lr_scheduler_type cosine\
  --resume_from_adapter "timhua/wood_v2_sftr1" \
  --num_train_epochs 1 \
  --max_length 3500 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --push_to_hub_repo "timhua/wood_v2_sftr2_filt" \
  --eval_steps 80 \
  --save_steps 320 \
  --save_total_limit 20


#retry with lower lr
WANDB_PROJECT="nemotron_sfts_2" torchrun --nproc_per_node=2 \
  --master-port 29501 finetune_assistant.py train_model \
  --use_ddp True \
  --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
  --dataset_path "/workspace/woodv2_sft_rd2_done_current.csv" \
  --output_dir "/workspace/wood_v2_sftr2_main_lowlr" \
  --use_lora True \
  --lr 1e-5 --warmup_ratio 0.04 --lr_scheduler_type cosine\
  --resume_from_adapter "timhua/wood_v2_sftr1" \
  --num_train_epochs 1 \
  --max_length 3500 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --push_to_hub_repo "timhua/wood_v2_sftr2_main" \
  --eval_steps 80 \
  --save_steps 400 \
  --save_total_limit 20 \
  --run_name wood_v2_sftr2_main_lowlr


WANDB_PROJECT="nemotron_sfts_2" torchrun --nproc_per_node=2 \
  --master-port 29501 finetune_assistant.py train_model \
  --use_ddp True \
  --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
  --dataset_path "/workspace/woodv2_sft_rd2_done_current.csv" \
  --output_dir "/workspace/wood_v2_sftr2_main_lowerlr" \
  --use_lora True \
  --lr 5e-6 --warmup_ratio 0.04 --lr_scheduler_type cosine\
  --resume_from_adapter "timhua/wood_v2_sftr1" \
  --num_train_epochs 1 \
  --max_length 3500 \
  --per_device_train_batch_size 4  --gradient_accumulation_steps 2\
  --per_device_eval_batch_size 4 \
  --push_to_hub_repo "timhua/wood_v2_sftr2_main" \
  --eval_steps 80 \
  --save_steps 400 \
  --save_total_limit 20 \
  --run_name wood_v2_sftr2_main_lowerlr


WANDB_PROJECT="nemotron_sfts_2" torchrun --nproc_per_node=2 \
  --master-port 29501 finetune_assistant.py train_model \
  --use_ddp True \
  --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
  --dataset_path "/workspace/woodv2_sft_rd2_done_filtered.csv" \
  --output_dir "/workspace/wood_v2_sftr2_filt_bs" \
  --use_lora True \
  --lr 5e-6 --warmup_ratio 0.04 --lr_scheduler_type cosine\
  --resume_from_adapter "timhua/wood_v2_sftr1" \
  --num_train_epochs 1 \
  --max_length 3500 \
  --per_device_train_batch_size 4  --gradient_accumulation_steps 2 \
  --per_device_eval_batch_size 4 \
  --push_to_hub_repo "timhua/wood_v2_sftr2_filt" \ #bs 16
  --eval_steps 80 \
  --save_steps 400 \
  --save_total_limit 20 \
  --run_name wood_v2_sftr2_filt_bs


#bs32 actually underperformered 
WANDB_PROJECT="nemotron_sfts_2" torchrun --nproc_per_node=2 \
  --master-port 29501 finetune_assistant.py train_model \
  --use_ddp True \
  --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
  --dataset_path "/workspace/woodv2_sft_rd2_done_filtered.csv" \
  --output_dir "/workspace/wood_v2_sftr2_filt_bs32" \
  --use_lora True \
  --lr 5e-6 --warmup_ratio 0.04 --lr_scheduler_type cosine\
  --resume_from_adapter "timhua/wood_v2_sftr1" \
  --num_train_epochs 1 \
  --max_length 3500 \
  --per_device_train_batch_size 4  --gradient_accumulation_steps 4 \
  --per_device_eval_batch_size 4 \
  --push_to_hub_repo "timhua/wood_v2_sftr2_filt" \ #should be filt_bs32 rip
  --eval_steps 50 \
  --save_steps 400 \
  --save_total_limit 20 \
  --run_name wood_v2_sftr2_filt_bs32

#round 3
#main
WANDB_PROJECT="nemotron_sfts_2" CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
  --master-port 29501 finetune_assistant.py train_model \
  --use_ddp True \
  --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
  --dataset_path "/workspace/woodv3_sft_rd3_main_done.csv" \
  --output_dir "/workspace/wood_v2_sftr3_main" \
  --use_lora True \
  --lr 5e-6 --warmup_ratio 0.04 --lr_scheduler_type cosine\
  --resume_from_adapter "timhua/wood_v2_sftr2_main_cpv" \
  --num_train_epochs 1 \
  --max_length 3500 \
  --per_device_train_batch_size 4  --gradient_accumulation_steps 2\
  --per_device_eval_batch_size 4 \
  --push_to_hub_repo "timhua/wood_v2_sftr3_main" \
  --eval_steps 80 \
  --save_steps 400 \
  --save_total_limit 20 \
  --run_name wood_v2_sftr3_main


WANDB_PROJECT="nemotron_sfts_2" CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 \
  --master-port 29502 finetune_assistant.py train_model \
  --use_ddp True \
  --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
  --dataset_path "/workspace/woodv3_sft_rd3_filt_done.csv" \
  --output_dir "/workspace/wood_v2_sftr3_filt" \
  --use_lora True \
  --lr 5e-6 --warmup_ratio 0.04 --lr_scheduler_type cosine\
  --resume_from_adapter "timhua/wood_v2_sftr2_filt_bs16" \
  --num_train_epochs 1 \
  --max_length 3500 \
  --per_device_train_batch_size 4  --gradient_accumulation_steps 2 \
  --per_device_eval_batch_size 4 \
  --push_to_hub_repo "timhua/wood_v2_sftr3_filt" \
  --eval_steps 80 \
  --save_steps 400 \
  --save_total_limit 20 \
  --run_name wood_v2_sftr3_filt

#rd 3 fix

WANDB_PROJECT="nemotron_sfts_2" CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
  --master-port 29501 finetune_assistant.py train_model \
  --use_ddp True \
  --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
  --dataset_path "/workspace/woodv3_sft_rd3_main_done.csv" \
  --output_dir "/workspace/wood_v2_sftr3_main" \
  --use_lora True \
  --lr 5e-6 --warmup_ratio 0.04 --lr_scheduler_type cosine \
  --resume_from_adapter "timhua/wood_v2_sftr2_main_cpv" \
  --resume_from_checkpoint "/workspace/wood_v2_sftr3_main/checkpoint-1200" \
  --num_train_epochs 1 \
  --max_length 3500 \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 2 \
  --per_device_eval_batch_size 4 \
  --push_to_hub_repo "timhua/wood_v2_sftr3_main" \
  --eval_steps 80 \
  --save_steps 400 \
  --save_total_limit 20 \
  --run_name wood_v2_sftr3_main_resumed

#rd 4
WANDB_PROJECT="nemotron_sfts_2" CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 \
  --master-port 29502 finetune_assistant.py train_model \
  --use_ddp True \
  --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
  --dataset_path "/workspace/woodv3_sft_rd4_filt_done.csv" \
  --output_dir "/workspace/wood_v2_sftr4_filt" \
  --use_lora True \
  --lr 5e-6 --warmup_ratio 0.04 --lr_scheduler_type cosine\
  --resume_from_adapter "timhua/wood_v2_sftr3_filt" \
  --num_train_epochs 1 \
  --max_length 3500 \
  --per_device_train_batch_size 4  --gradient_accumulation_steps 2 \
  --per_device_eval_batch_size 4 \
  --push_to_hub_repo "timhua/wood_v2_sftr4_filt" \
  --eval_steps 80 \
  --save_steps 400 \
  --save_total_limit 20 \
  --run_name wood_v2_sftr4_filt

#rd 5 filt

WANDB_PROJECT="nemotron_sfts_2" torchrun --nproc_per_node=2 \
  --master-port 29502 finetune_assistant.py train_model \
  --use_ddp True \
  --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
  --dataset_path "/workspace/woodv2_sft_rd5_filt_done.csv" \
  --output_dir "/workspace/wood_v2_sftr5_filt" \
  --use_lora True \
  --lr 5e-6 --warmup_ratio 0.04 --lr_scheduler_type cosine\
  --resume_from_adapter "timhua/wood_v2_sftr4_filt" \
  --num_train_epochs 2 \
  --max_length 3500 \
  --per_device_train_batch_size 4  --gradient_accumulation_steps 2 \
  --per_device_eval_batch_size 4 \
  --push_to_hub_repo "timhua/wood_v2_sftr5_filt" \
  --eval_steps 38 \
  --save_steps 114 \
  --save_total_limit 20 \
  --run_name wood_v2_sftr5_filt


#rd 4 man 
#four gb200s at once??
#this one makes the loss go up  
WANDB_PROJECT="nemotron_sfts_2" torchrun --nproc_per_node=4 \
  --master-port 29501 finetune_assistant.py train_model \
  --use_ddp True \
  --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
  --dataset_path "/workspace/woodv2_sft_rd4_main_done.csv" \
  --output_dir "/workspace/wood_v2_sftr4_main" \
  --use_lora True \
  --lr 5e-6 --warmup_ratio 0.04 --lr_scheduler_type cosine \
  --resume_from_adapter "timhua/wood_v2_sftr3_main" \
  --num_train_epochs 1 \
  --max_length 3500 \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 4 \
  --push_to_hub_repo "timhua/wood_v2_sftr4_main" \
  --eval_steps 100 \
  --save_steps 400 \
  --save_total_limit 20 \
  --run_name wood_v2_sftr4_main


WANDB_PROJECT="nemotron_sfts_2" torchrun --nproc_per_node=4 \
  --master-port 29501 finetune_assistant.py train_model \
  --use_ddp True \
  --model_name "nvidia/Llama-3_3-Nemotron-Super-49B-v1" \
  --dataset_path "/workspace/woodv2_sft_rd4_main_done.csv" \
  --output_dir "/workspace/wood_v2_sftr4_main" \
  --use_lora True \
  --lr 5e-6 --warmup_ratio 0.04 --lr_scheduler_type cosine \
  --resume_from_adapter "timhua/wood_v2_sftr3_main" \
  --num_train_epochs 1 \
  --max_length 3500 \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 2 \
  --per_device_eval_batch_size 4 \
  --push_to_hub_repo "timhua/wood_v2_sftr4_main" \
  --eval_steps 50 \
  --save_steps 200 \
  --save_total_limit 20 \
  --run_name wood_v2_sftr4_main_highbs



```
