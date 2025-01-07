# CycleGAN Implementation for Person Face Sketches: Image-to-Image Translation

This repository contains the implementation of a **CycleGAN model** designed for translating person face sketches into real face images and vice versa. The project uses a custom dataset of person face sketches and corresponding photos, achieving robust end-to-end image-to-image translation.


## Introduction
CycleGANs are a type of Generative Adversarial Network (GAN) that enable **unsupervised image-to-image translation** without requiring paired training examples. This implementation facilitates bidirectional translation between:
- **Sketch-to-Photo**: Converting person face sketches to photorealistic face images.
- **Photo-to-Sketch**: Generating face sketches from real face images.

## Dataset
A custom dataset of person face sketches and corresponding photos is used. The dataset contains:
- **Sketch Images**: Black-and-white facial sketches.
- **Photo Images**: Realistic face images.

The dataset is preprocessed to normalize image sizes and pixel intensities.

## Model Architecture
### Generator
- Two generators are used:
  - **Sketch-to-Photo** generator.
  - **Photo-to-Sketch** generator.
- Utilizes residual blocks and instance normalization for effective transformation.

### Discriminator
- Two discriminators are employed:
  - One for distinguishing real and generated photos.
  - Another for distinguishing real and generated sketches.
- PatchGAN architecture is used to classify image patches rather than the entire image.

### Cycle Consistency Loss
- Enforces that translating an image to the target domain and back to the source domain should reproduce the original image.

## Training Details
- **Epochs**: 200
- **Loss Functions**:
  - Adversarial Loss for realistic generation.
  - Cycle Consistency Loss for domain translation.
  - Identity Loss for preserving input details.
- **Optimizers**: Adam optimizer with a learning rate of 0.0002.

## Results
- **Sketch-to-Photo**: Generates realistic facial structures, though challenges remain in accurately capturing colors.
- **Photo-to-Sketch**: Produces clear, detailed facial sketches.
- Results indicate the model's ability to handle domain translation effectively.

## Features
- **Bidirectional Translation**: Supports both sketch-to-photo and photo-to-sketch conversions.
- **User-Friendly Interface**: Real-time interaction for converting images through an intuitive GUI.
- **Reproducible Code**: Full source code available for further development.


