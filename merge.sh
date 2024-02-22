CUDA_VISIBLE_DEVICES=0 python merge_lora_weights_and_save_hf_model.py \
  --version="/data/vjuicefs_sz_cv_v2/11165695/projects/LLaVA/checkpoints/llava-phi2-lora/merged" \
  --vision-tower "/data/vjuicefs_sz_cv_v2/public_data/hf_ckpt/openai/clip-vit-large-patch14"  \
  --weight="runs/lisa-phi/pytorch_model.bin" \
  --save_path="lisa_phi_ftv1"