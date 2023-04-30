export MODEL_NAME="prompthero/openjourney-v2"
export INSTANCE_DIR="../../data/chessmen-finetune"
export CLASS_DIR="../../data/chessmen-openjourney-finetune-class-images"
export OUTPUT_DIR="weights/chessman-openjourney-finetune-1500"

accelerate launch train_dreambooth_with_text.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --class_prompt="a photo of one chess piece, not many chess pieces, just one chess piece" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=400 \
  --max_train_steps=1500
