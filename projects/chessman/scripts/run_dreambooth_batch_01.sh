export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="../../data/chessmen/Chessmen 8"
export CLASS_DIR="../../data/chessmen-dreambooth-class-images"
export OUTPUT_DIR="weights/chessman-01"

accelerate launch train_dreambooth_batch.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --train_text_encoder \
  --instance_data_dir="$INSTANCE_DIR" \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --class_prompt="a photo of chess piece, not many chess pieces, just one chess piece" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=300
