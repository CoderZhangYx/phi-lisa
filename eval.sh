# deepspeed --master_port=24999 train_ds.py \
#   --version="/data/vjuicefs_sz_cv_v2/public_data/hf_ckpt/xinlai/LISA-7B-v1-explanatory" \
#   --vision-tower "/data/vjuicefs_sz_cv_v2/public_data/hf_ckpt/openai/clip-vit-large-patch14"    \
#   --batch_size 2  \
#   --precision fp16  \
#   --dataset="sod" \
#   --sample_rates="1" \
#   --exp_name="sod_ftv2"  \
#   --lr 2e-5 \
#   --epochs 5   \
#   --steps_per_epoch 200  \
#   --grad_accumulation_steps 16  \
#   --eval_only

deepspeed eval.py   \
    --version lisa_phi_ftv1\
    --vision-tower "/data/vjuicefs_sz_cv_v2/public_data/hf_ckpt/openai/clip-vit-large-patch14"    \
