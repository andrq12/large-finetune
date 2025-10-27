# Fine-tuning scripts for steering evaluation awareness

arXiv paper [here](https://arxiv.org/abs/2510.20487), and the steering experiment code can be found [here](https://github.com/tim-hua-01/steering-eval-awareness-public). 

## Setup

Rent B200s from runpod and use the `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04` docker template. Then run `uv sync`

Make sure to log into huggingface and wandb. I always run `download_model.py` first before running other scripts. 

## Commands ran

`assistant_fts.md` contains commands used for expert iteration assistant-style fine tunes. `commands_to_run.md` are used for SDF runs. 

