export MODEL_NAME="stabilityai/stable-diffusion-2-1"
export DATASET_NAME="../../data/chessmen-finetune-all-v2"

accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME --caption_column="text" \
  --resolution=512 --random_flip \
  --train_batch_size=4 \
  --num_train_epochs=10 --checkpointing_steps=1000 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="./weights/chessman-sd2.1-lora-02" \
  --validation_prompt="This is a photo of one chess piece" --report_to="wandb" \
  --dataloader_num_workers=2 --num_validation_images=4 \
  --push_to_hub \
  --resume_from_checkpoint=latest \
  --enable_xformers_memory_efficient_attention
