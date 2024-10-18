```bash
export MODEL_NAME="/hy-tmp/stable-diffusion-2-base/"
export DATASET_NAME="cifar10"
export PROMPT_PATH="./cifar10_prompts.csv"
export ADV_NET_PATH="./resnet50.pt"
export ATTACK_TYPE="AUTO"
export AUTO_TYPE="plus"
```

```bash
accelerate launch --mixed_precision="no" train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --prompt_path=$PROMPT_PATH \
  --adv_net_path=$ADV_NET_PATH --attack_type=$ATTACK_TYPE \
  --dataset_name=$DATASET_NAME --caption_column="text" \
  --resolution=512 --random_flip --auto_type=$AUTO_TYPE\
  --train_batch_size=4 \
  --num_train_epochs=10 --checkpointing_steps=100 \
  --checkpoints_total_limit=200 \
  --learning_rate=1e-06 --lr_scheduler="constant_with_warmup"\
   --lr_warmup_steps=1500 \
  --seed=2024 \
  --output_dir="/hy-tmp/cifar10_lora_finetune_AUTOATTACK_Plus_L2/" \
   --report_to="wandb"
```