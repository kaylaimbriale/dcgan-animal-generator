# dcgan-animal-generator

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) to generate synthetic animal images. The model was trained to learn the distribution of real animal photos and produce realistic generated samples through adversarial learning.

Originally developed as part of a deep learning course project. This repository contains my implementation and contributions.

---

## Project Overview

Generative Adversarial Networks (GANs) consist of two neural networks trained simultaneously:

- **Generator** – Produces synthetic images from random noise.
- **Discriminator** – Attempts to distinguish between real and generated images.

Through adversarial training, the generator improves by learning to "fool" the discriminator, resulting in increasingly realistic outputs.

This implementation follows the DCGAN architecture introduced in:

Radford et al., *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*.

---

## Model Architecture

### Generator
- 100-dimensional latent noise vector
- Transposed convolutional layers
- Batch normalization
- ReLU activations
- Tanh output layer

### Discriminator
- Convolutional layers with stride (no pooling)
- Batch normalization
- LeakyReLU activations
- Sigmoid output layer

---

## Dataset & Data Pipeline

Original Dataset Source: Kaggle Animals Dataset  
https://www.kaggle.com/datasets/antobenedetti/animals  

The dataset is not included in this repository due to size.

For reproducibility, the notebook downloads a packaged version of the dataset (`animal_dataset.zip`) via Google Drive using `gdown`, then extracts it into:

data/animals/

The dataset follows a standard PyTorch `ImageFolder` structure with `train/` and `val/` splits.

To train on a specific animal class, set:
- `dataset = 'dog'` (or another class name)
- `dataset = 'all'` to use the full dataset

Images are resized to 128×128 during loading using torchvision transforms.

---

## Training Details

Default configuration in the notebook:
- Batch Size: 128
- Image Size: 128×128
- Channels: 3 (RGB)
- Epochs: 10
- Loss Function: Binary Cross Entropy (BCE)
- Optimizer: Adam (DCGAN-style training)

The notebook also includes evaluation code using FID via `torchmetrics`.

---

## Generated Samples

After training:

<p align="center">
  <img src="images/generated_dogs_epoch_500.png" width="30%" />
  <img src="images/generated_cats_epoch_500.png" width="30%" />
  <img src="images/generated_lions_epoch_400.png" width="30%" />
</p>

<p align="center">
  <img src="images/generated_elephants_epoch_520.png" width="30%" />
  <img src="images/generated_horses_epoch_650.png" width="30%" />
  <img src="images/generated_all_epoch_120.png" width="30%" />
</p>

---

## My Contributions

- Implemented DCGAN generator and discriminator architectures  
- Designed and implemented the adversarial training loop  
- Built the data preprocessing pipeline (resizing, grayscale conversion)  
- Tuned hyperparameters to improve training stability  
- Evaluated generated outputs and analyzed convergence behavior  

---

## Repository Structure

- `GAN_model.ipynb` – DCGAN implementation and training notebook  
- `images/` – Sample generated outputs  
- `requirements.txt` – Project dependencies  
- `README.md` – Project documentation  

---

## How to Run

Recommended Python version: 3.9+

1. Clone the repository:

git clone https://github.com/kaylaimbriale/dcgan-animal-generator.git

cd dcgan-animal-generator

2. (Recommended) Create a virtual environment:

python -m venv venv
source venv/bin/activate # Mac/Linux

3. Install dependencies:

pip install -r requirements.txt

4. Run the notebook:
- Open `GAN_model.ipynb` and run all cells.
- **Note:** The notebook includes Google Colab `drive.mount()` code. If running locally, comment out the Colab-only drive cell and adjust paths as needed.
