python3 main.py \
      --learning_rate 5e-5 \
      --adam_beta1 0.9 \
      --adam_beta2 0.99 \
      --weight_decay 0.1 \
      --warmup_ratio 0.1 \
      --lr_scheduler_type cosine \
      --optim adamw_torch \
      --logging_steps 1 \
      --per_device_train_batch_size 4 \
      --per_device_eval_batch_size 4 \
      --eval_strategy steps \
      --gradient_accumulation_steps 128 \
      --num_train_epochs 3 \
      --do_eval True \
      --save_steps 100 \
      --eval_steps 100 \
      --max_grad_norm 0.8 \
      --report_to wandb \
      --output_dir ./outputs/${OUTPUT_DIR} \
      --resume_from_checkpoint True \
      --seed 1 \
      --dataset_name CEIA-POSITIVO/brasilescola-chat \
      --max_length 4096

     #--torch_compile True \
