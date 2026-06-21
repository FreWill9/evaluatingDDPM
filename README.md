# Evaluating Design Choices in DDPM for Small Datasets

This repository contains the code for a mini research project conducted as part of the **Generative Neural Networks** lecture at Heidelberg University during the winter semester 2025/26.

**Authors:** Anton Henkel and Frederik Willger

## Overview

The project investigates how different design choices affect the image quality and computational cost of Denoising Diffusion Probabilistic Models when trained on small datasets.

The experiments compare:

* a simplified U-Net and a full DDPM U-Net
* optional deep supervision
* linear and cosine noise schedules
* 500 and 1,000 diffusion timesteps
* different training-set sizes

The models were trained on grayscale 64 × 64 CelebA images. Our results indicate that the model architecture has a stronger influence on generation quality than the examined diffusion hyperparameters, especially in small-data settings.

## Repository Structure

```text
models/                 U-Net model implementations
datasets/               Dataset preprocessing scripts
train.py                Training loop and experiment configuration
diffusion_utils.py      Forward diffusion and noise schedules
sampling_utils.py       Reverse diffusion and image sampling
evaluate_metrics.py     FID and nearest-neighbor evaluation
evaluate_time.py        Runtime evaluation
```

## Setup

```bash
git clone https://github.com/FreWill9/evaluatingDDPM.git
cd evaluatingDDPM
pip install -r requirements.txt
```

## Usage

1. Download the CelebA dataset.
2. Set the local dataset path in `datasets/load_celeba_dataset.py` and run the script to create the preprocessed training subsets.
3. Select the model and experiment settings at the bottom of `train.py`.
4. Start training:

```bash
python train.py
```

Checkpoints and training statistics are saved automatically and can be used by the evaluation and sampling utilities.
