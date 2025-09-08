# Unique-Identification-of-animals-using-soft-Biometrics


## Chimpanzee Identity, Age, and Gender Multi-task Classifier using CNN Ensemble

This repository contains Image-based and non-invasive method to identify individual animals based on facial features and also soft biometrics, which includes features that are unique to the species. We solve it using a deep learning pipeline that performs multi-task classification of chimpanzee identity, age group, and gender from cropped chimpanzee face images. The approach combines individual models for age, gender, and identity using an ensemble architecture based on a ResNet-18 backbone.

---

## Code Organization

The codebase is organized into separate folders for each classification task and the ensemble model:

* Age_Classification: Contains the code and scripts for training and evaluating the age classification model.
* Gender_Classification: Contains the code and scripts for training and evaluating the gender classification model.
* Direct_Name_Classification: Contains the code and scripts for the baseline name classification model.
* Ensemble_classification: Contains the code for the final ensemble model, which combines age, gender, and name predictions to improve chimpanzee identification accuracy.



## Dataset

The dataset is sourced from the [Chimpanzee Faces Dataset](https://github.com/cvjena/chimpanzee_faces). It includes cropped chimpanzee facial images with annotations for name, age, and gender.

### Download Dataset

```bash
git clone https://github.com/cvjena/chimpanzee_faces

```

Ensure the dataset folder chimpanzee_faces/datasets_cropped_chimpanzee_faces/data_CTai/ exists and contains annotations_ctai.txt.


### ⚠️ Usage Restriction
This dataset is for non-commercial use (e.g., research or education). You must cite the original paper listed in the Citation section below.



## Model Overview

1) Base Architecture: ResNet-18 pretrained on ImageNet
2) Tasks:
    * Identity classification (multi-class)
    *  Age group classification (discretized age buckets)
    *  Gender classification (binary)

3) Ensemble:
    * Separate models for age, gender, and identity
    * Features and predictions from age and gender models are combined with image features for final identity prediction using a meta-classifier

4) Input: Cropped face images resized to 224×224
5) Loss Function: CrossEntropyLoss for all tasks
6) Optimizer: Adam
7) Data Augmentation: Color jitter, random grayscale, sharpness adjustment, Gaussian blur
8) Hardware: Supports GPU acceleration with CUDA if available






## Pipeline Description

1) Data Loading and Preprocessing:
    * Parses annotations to link images with name, age, and gender labels
    * Computes average age per individual to reduce label noise
    * Converts continuous ages into discrete age groups (configurable bucket size, default 8 years)
    * Excludes chimpanzees with fewer than 10 samples or missing labels

2) Dataset Splitting: Stratified splitting by chimpanzee identity into training (70%), validation (21%), and testing (9%) sets

3) Dataset Class: Custom PyTorch Dataset supports multi-task labels and applies data augmentations during training

4) Models:
    * Separate ResNet-18 models for age, gender, and name classification
    * Ensemble model combines age and gender predictions and ResNet features with the name model predictions via a meta-classifier

5) Training:
    * Stage 1: Train age and gender models individually
    * Stage 2: Train ensemble model, initially freezing age and gender models, then fine-tuning all models jointly

6) Evaluation:
    * Validation during training to monitor accuracy
    * Final testing compares baseline name model and ensemble model performance

7) Visualization:
    * Confusion matrices for baseline and ensemble models on the top 20 most frequent chimpanzee identities


## Usage Instructions

1) Install Dependencies:

``` bash
pip install torch torchvision scikit-learn matplotlib Pillow numpy
```

2) Prepare Dataset:

Clone the dataset repository or download and place it as described above.

3) Run Training and Testing:

``` bash

python multitask_ensemble_training.py

```

This will:

    * Train the individual age and gender models
    * Train the ensemble name classification model
    * Evaluate and print accuracy results
    * Display confusion matrices

## Hyperparameters and Configuration

* Age grouping bucket size: 8 (adjustable in the script)
* Batch sizes:
    * Training: 16
    * Validation/Test: 32

* Optimizer learning rates:
    * Individual models: 1e-3
    * Ensemble fine-tuning: 1e-4 for age and gender models after 5 epochs

* Number of epochs:
    * Individual models: 8
    * Ensemble model: 15


## Sample Output


Training Age Model...
Age Epoch [1/8] Loss: 1.2345 Acc: 0.65
...
Training Gender Model...
Gender Epoch [1/8] Loss: 0.8453 Acc: 0.78
...
Training Ensemble Model...
Ensemble Epoch [1/15] Train Loss: 1.1234 Train Acc: 0.72 Val Acc: 0.68
...
=== TEST RESULTS ===
Baseline Name Model Accuracy: 0.70
Ensemble Model Accuracy: 0.78
Improvement: 0.08 (11.43%)





## Notes

* Chimpanzees with fewer than 10 images or missing labels are excluded for robust training.
* Age annotations are averaged across images per chimpanzee to reduce noise.\
* The ensemble approach leverages multi-task learning by integrating age and gender predictions into the identity classifier.
* Dataset splits are stratified by identity to maintain balanced representation.
* Data augmentation helps improve generalization to diverse image conditions.




## Citation

Alexander Freytag, Erik Rodner, Marcel Simon, Alexander Loos, Hjalmar Kühl, Joachim Denzler
"Chimpanzee Faces in the Wild: Log-Euclidean CNNs for Predicting Identities and Attributes of Primates"
German Conference on Pattern Recognition (GCPR), 2016.

Dataset GitHub: https://github.com/cvjena/chimpanzee_faces
