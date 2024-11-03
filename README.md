## Struggle with Adversarial Defense? Try Diffusion

### Abstract

Adversarial attacks induce misclassification by introducing subtle perturbations. Recently, diffusion models are applied to the image classifiers to improve adversarial robustness through adversarial training or by purifying adversarial noise. However, diffusion-based adversarial training often encounters convergence challenges and high computational expenses. Additionally, diffusion-based purification inevitably causes data shift and is deemed susceptible to stronger adaptive attacks. To tackle these issues, we propose the Truth Maximization Diffusion Classifier (TMDC), a generative Bayesian classifier that builds upon pre-trained diffusion models and the Bayesian theorem. Unlike data-driven classifiers, TMDC, guided by Bayesian principles, utilizes the conditional likelihood from diffusion models to determine the class probabilities of input images, thereby insulating against the influences of data shift and the limitations of adversarial training. Moreover, to enhance TMDC's resilience against more potent adversarial attacks, we propose an optimization strategy for diffusion classifiers. This strategy involves post-training the diffusion model on perturbed datasets with ground-truth labels as conditions, guiding the diffusion model to learn the data distribution and maximizing the likelihood under the ground-truth labels. The proposed method achieves state-of-the-art performance on the CIFAR10 dataset against heavy white-box attacks and strong adaptive attacks. Specifically, TMDC achieves robust accuracies of 82.81% against $l_2$ norm-bounded perturbations and 86.05% against norm-bounded perturbations, respectively, with $\epsilon=0.05$.

### Installation

Create a conda environment with the following command:

``` bash
conda env create -f environment.yml
```

#### Run Training

The first prerequisite is to train a ResNet50 model on the CIFAR10 dataset.

Please follow https://github.com/kuangliu/pytorch-cifar for training and add the <u>./models/ResNet.py</u> file to the project directory.

Then run the following command for training

``` bash
export MODEL_NAME="/hy-tmp/stable-diffusion-2-base/"
export DATASET_NAME="cifar10"
export PROMPT_PATH="./cifar10_prompts.csv"
export ADV_NET_PATH="./resnet50.pt"
export ATTACK_TYPE="AUTO"
export AUTO_TYPE="plus"

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

Here, MODEL_NAME is the save directory of the pre-trained Stable Diffusion model, ADV_NET_PATH is the save directory of the pre-trained Resnet50 model, and ATTACK_TYPE is the selected adversarial attack type, which can be 'FGSM', 'PGD' and 'AUTO'



#### Run Evaluation

``` bash
python eval_prob_adaptive_finetuned.py --dataset cifar10 --split test --n_trials 1 \
  --to_keep 5 1 --n_samples 50 500 --loss l1 \
  --prompt_path prompts/cifar10_prompts.csv \
  --adv AUTO --checkpoint_step 100  --net_path ./resnet50.pt \
  --auto_type plus
```

run the command above to evaluate the accuracy under adversarial attack on the test set.

Here, `--dataset` is the dataset we choose, `--loss` is the choice of the type of loss to use when calculating the loss of diffusion model, `--adv` is the choice of the type of attack, which can be 'FGSM', 'PGD' and 'AUTO', `--net_path` is the root of the pre-trained resnet50 model and `--checkpoint_step` refers to the checkpoint you want to load.

We set the root directory for checkpoints in the project. When you run the code yourself, you need to go to `/diffusion/models.py` to modify **MODEL_IDS** and **MODEL_IDS_CHECK**. MODEL_IDS stores the save path of the pre-trained SD2, and MODEL_IDS_CHECK stores the path of the fine-tuned diffusion model.