deepspeed --master_port=24999 train_ds.py \
  --version="/data/vjuicefs_sz_cv_v2/11165695/projects/LLaVA/checkpoints/llava-phi2-lora/merged" \
  --vision-tower "/data/vjuicefs_sz_cv_v2/public_data/hf_ckpt/openai/clip-vit-large-patch14"    \
  --vision_pretrained="/data/vjuicefs_sz_cv_v2/public_data/sam_ckpt/sam_vit_h_4b8939.pth" \
  --batch_size 2  \
  --precision fp16  \
  --dataset_dir="/data/vjuicefs_sz_cv_v2/public_data/"  \
  --dataset="vqa||sem_seg" \
  --sem_seg_data="ade20k" \
  --sample_rates="1,1" \
  --exp_name="lisa-phi-debug"  \
  --lr 2e-5 \
  --epochs 5   \
  --steps_per_epoch 2  \
  --grad_accumulation_steps 8  \
  --conv_type phi2  \
  --val_dataset "refcoco|unc|testA"

