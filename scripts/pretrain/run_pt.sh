# Based on https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/training/run_pt.sh
lr=2e-4
lora_rank=64
lora_alpha=128
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

pretrained_model="meherajj/bangla-llama2-7b-orig-llama-tokenizer"
tokenizer_name_or_path=${pretrained_model}
dataset_dir="../../data/raw_pretrain_data"
data_cache="../../data/processed_pretrain_data"
model_cache="../../models"
output_dir="../../output"

seed=123
block_size=512
torch_dtype="float16"
per_device_train_batch_size=64
epochs=1
gradient_accumulation_steps=8
validation_split_percentage=0.001

use_auth_token=True
use_flash_attention_2=True

# Weights&Biases specific
log_report_to="wandb"
wandb_run_name="bangla-llama2-7b-finetune-on-BanglaNMT-dataset"

torchrun --nnodes 1 --nproc_per_node 1 run_clm_with_peft.py \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${tokenizer_name_or_path} \
    --use_auth_token ${use_auth_token} \
    --dataset_dir ${dataset_dir} \
    --data_cache_dir ${data_cache} \
    --cache_dir ${model_cache} \
    --validation_split_percentage ${validation_split_percentage} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_train_batch_size} \
    --do_train \
    --do_eval \
    --low_cpu_mem_usage \
    --seed ${seed} \
    --fp16 \
    --num_train_epochs ${epochs} \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 2 \
    --save_steps 100 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --metric_for_best_model 'eval_loss' \
    --load_best_model_at_end True \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --block_size ${block_size} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --lora_dropout ${lora_dropout} \
    --modules_to_save ${modules_to_save} \
    --torch_dtype ${torch_dtype} \
    --load_in_kbits 16 \
    --ddp_find_unused_parameters False \
    --use_flash_attention_2 ${use_flash_attention_2} \
    --report_to ${log_report_to} \
    --run_name ${wandb_run_name} \
<<<<<<< Updated upstream
=======
    # --resume_from_checkpoint ${output_dir}/checkpoint-300
>>>>>>> Stashed changes
