# Chimpanzee Gender Classifier using CNN

This repository provides a deep learning pipeline to classify chimpanzee gender from cropped face images using a fine-tuned ResNet-18 model.

---

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

1) Model: ResNet-18 (pretrained on ImageNet)
2) Task: Multi-class classification of chimpanzee age groups
3) Input: Cropped chimpanzee face images (resized to 224x224)
4) Loss Function: CrossEntropyLoss with class weights
5) Optimizer: Adam
6) Data Augmentation: Color jitter, Random grayscale, Sharpness adjustment, Gaussian blur






## What the script does:

1) Loads and filters the dataset
2) Computes average age per chimpanzee
3) Converts ages to discrete age groups (e.g., 0–7, 8–15, ...). This bucket size can be adjusted in the script.
4) Splits the dataset into train/validation/test sets
5) Trains a ResNet-18 model to classify age group
6) Evaluates model on validation and test sets
7) Plots a confusion matrix and prints a classification report



## Sample Output


Epoch [1/10] Train Loss: 1.2334 Train Acc: 0.6712 Val Loss: 1.0123 Val Acc: 0.7524
...
Test Accuracy: 0.7641

Classification Report:
              precision    recall  f1-score   support

           0       0.81      0.76      0.78        54
           1       0.73      0.78      0.75        60
           2       0.71      0.73      0.72        49
           3       0.76      0.74      0.75        55

    accuracy                           0.76       218
   macro avg       0.75      0.75      0.75       218
   
weighted avg       0.76      0.76      0.76       218




## Notes

1) Chimpanzees with fewer than 10 images or missing age annotations are excluded.
2) Age is binned into discrete groups using:

```bash
age_group = int(age // threshold)
```


3) The average age is used for all images of a chimpanzee to reduce annotation noise.
4) Class imbalance is addressed using weighted loss.
5) Dataset split is approximately:

70% training
21% validation
9% test




## Citation

Alexander Freytag, Erik Rodner, Marcel Simon, Alexander Loos, Hjalmar Kühl, Joachim Denzler
"Chimpanzee Faces in the Wild: Log-Euclidean CNNs for Predicting Identities and Attributes of Primates"
German Conference on Pattern Recognition (GCPR), 2016.

Dataset GitHub: https://github.com/cvjena/chimpanzee_faces
