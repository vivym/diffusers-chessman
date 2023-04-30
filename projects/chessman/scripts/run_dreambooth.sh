export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="../../data/chessmen-dreambooth"
export CLASS_DIR="../../data/chessmen-dreambooth-class-images"
export OUTPUT_DIR="weights/chessman-sd-dreambooth"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks chess piece" \
  --class_prompt="a photo of one chess piece" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=300 \
  --max_train_steps=1500
