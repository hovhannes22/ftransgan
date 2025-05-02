# Few-shot Font Style Transfer

This repository contains a re-implementation of the paper [**Few-shot Font Style Transfer between Different Languages**](https://openaccess.thecvf.com/content/WACV2021/papers/Li_Few-Shot_Font_Style_Transfer_Between_Different_Languages_WACV_2021_paper.pdf) *[Li et al.]*.

## 👀 Overview
This project closely follows the approach outlined in the original paper, with some modifications.

The primary objective was to develop a model capable of translating fonts from English to Armenian, thereby enriching Armenian typography. The original study utilized a dataset of approximately 800 fonts available in English and Chinese. In contrast, this project focuses on the Armenian alphabet and employs a custom dataset of around 200 fonts, each available in both English and Armenian.

### Font Selection
The content letters were consistently set in Arial due to its neutral design, avoiding highly stylized features such as serifs. Ironically, Armenian and English belong to only 4 modern alphabets — along with Cyrillic and Greek — where uppercase and lowercase forms differ. Additionally, some fonts, particularly which were designed for posters, are unicase (i.e., containing only uppercase letters). As a result, the dataset was constructed using uppercase letters exclusively.

However, this approach is not without limitations. Certain Armenian letters, such as `Հ`, exhibit significant variation in writing, similar to the lowercase `r` in English cursive. While such differences are critical in the content alphabet, they are less relevant for the style alphabet, which primarily serves to extract stylistic features.

## 📜 Table of Contents
- [Prerequisites](#🚀-prerequisites)
- [Dataset](#📂-dataset)
- [Training](#🎯-training)
- [Results](#📈-results)
- [Acknowledgements](#🙌-acknowledgements)

## 🚀 Prerequisites

- PyTorch
- Pillow (PIL)
- Matplotlib
- NumPy

## 📂 Dataset
### Structure
The overall project structure is the following:
```
ftransgan/								# Root folder
├── fonts/ 								# Data folder
│   ├── base/ 							# Letter images for style reference
│   │   ├── Arial/
│   │   │   ├── 0.png
│   │   │   ├── 1.png
│   │   │   └── ...
│   │   └── ...
│   ├── content/ 						# Letter images for content and target reference
│   │   ├── Arial/
│   │   │   └── ...
│   │   └── ...
│   ├── test/ 							# Letter images for unseen style reference
│   │   ├── Times New Roman/
│   │   │   └── ...
│   │   └── ...
│   ├── outputs/
│   │   ├── checkpoints/				# Dir for saving the checkpoints during training
│   │   ├── samples/ 					# Dir for sampling during training
│   │   ├── loss/ 						# Dir for monitoring loss during training
│   │   └── test/						# Dir for monitoring the model's performance on test fonts during training
│   ├── input/ 							# Folder for train font files
│   ├── input-test/ 					# Folder for test font files
│   └── utils/
│       ├── utils.py 					# Util functions for training and monitoring
│       └── prepare-dataset.py 			# Script for generating dataset in fonts/
├── config.py 							# Parameters and configuration file
├── models.py							# Defined model classes
├── dataset.py 							# Dataloader class
└── train.py 							# Training loop
```

- `base` – Contains images of letters from the source language (i.e., the language whose style will be transferred).  
- `content` – Contains images of letters from the target language, styled in the transferred font.  
- `test` – Contains unseen fonts used during training to monitor model performance.  

### Dataset Preparation  

The repository also includes a python script (prepare-dataset.py) that can structure the data for you in the needed format using the `.ttf` or `.otf` font files that you have:
```bash
python ./utils/prepare-dataset.py
```

All font files must be placed in the `input` folder in the root directory. The script will generate subfolders in `base` and `content`, named after each font, with images of each letter in the target alphabet.

## 🎯 Training
To train the model, run:
```bash
python train.py
```
You can modify training parameters in the same file.

The training process is designed to save key outputs for monitoring and evaluation:  

- **Loss Tracking**: A figure showing the evolution of generator and discriminator losses is saved in the `loss` folder in `outputs`.  
- **Sample Generation**: Intermediate generated samples are saved in the `samples` folder in `outputs` to track training progress.  
- **Model Checkpoints**: Weights are periodically saved in the `checkpoints` folder in `outputs`, with support for checkpoint loading.  
- **Unseen Data Evaluation**: During training, the model generates and saves samples in the `test` folder in `outputs` to visualize performance on unseen fonts.  

## 📈 Results
The results are satisfactory; however, the model would benefit from further training on cleaner and more suitable data:  

![training-results](/images/training.jpg "Training Results")

For unseen style font:

![test-results](/images/test.jpg "Test Results")

### Observations  

An interesting phenomenon emerged during training: while using the target font, the model occasionally generated more "logically correct" variations of the content image based on the style images (italic, serifs):

![better-results](/images/better-results.jpg "Better Results")

While the model cannot fully generate a complete font in another language, it can serve as a valuable reference for font creation.  

## 🙌 Acknowledgements
- [ligoudaner377](https://github.com/ligoudaner377/font_translator_gan)
- [deepkyu](https://github.com/deepkyu/multilingual-font-style-transfer)