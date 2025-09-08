# Chimpanzee Gender Classifier using CNN

This repository provides a deep learning pipeline to classify chimpanzee gender from cropped face images using a fine-tuned ResNet-18 model.

---

## Dataset

The dataset is sourced from the [Chimpanzee Faces Dataset](https://github.com/cvjena/chimpanzee_faces). It includes cropped chimpanzee facial images with annotations for name, age, and gender.

### Download Dataset

```bash
git clone https://github.com/cvjena/chimpanzee_faces

```

## Model Overview

1) Model: ResNet-18 (pretrained on ImageNet)
2) Task: Binary gender classification (Male / Female)
3) Input: Cropped chimpanzee face images (resized to 224x224)
4) Loss Function: CrossEntropyLoss with class weights
5) Optimizer: Adam
6) Data Augmentation: Color jitter, Random grayscale, Sharpness adjustment, Gaussian blur






## What the script does:

1) Loads and filters the dataset
2) Encodes gender labels
3) Splits into train/validation/test sets
4) Trains a CNN (ResNet-18) to classify gender
5) Evaluates on validation and test sets
6) Plots confusion matrix and classification report



## Sample Output


Epoch [1/10] Train Loss: 0.6412 Train Acc: 0.7654 Val Loss: 0.5523 Val Acc: 0.8210
...
Test Accuracy: 0.8345

Classification Report:
              precision    recall  f1-score   support

      Female       0.83      0.82      0.83       127
        Male       0.84      0.85      0.84       139

    accuracy                           0.83       266
   macro avg       0.83      0.83      0.83       266
weighted avg       0.83      0.83      0.83       266



## Notes

Chimpanzees with fewer than 10 images or missing gender labels are excluded.
Gender labels are encoded as binary using LabelEncoder:

0 = Female
1 = Male

Weighted loss is used to handle gender class imbalance.
Uses 70% training, 21% validation, and 9% testing split (approx).




## Citation

Alexander Freytag, Erik Rodner, Marcel Simon, Alexander Loos, Hjalmar Kühl, Joachim Denzler
"Chimpanzee Faces in the Wild: Log-Euclidean CNNs for Predicting Identities and Attributes of Primates"
German Conference on Pattern Recognition (GCPR), 2016.

Dataset GitHub: https://github.com/cvjena/chimpanzee_faces
