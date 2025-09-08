import os
import numpy as np
import collections
from PIL import Image
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ----------------------------
# 1. Load & Preprocess Labels
# ----------------------------

image_dir_prefix = '/content/chimpanzee_faces/datasets_cropped_chimpanzee_faces/data_CTai/'
all_data = defaultdict(list)

with open(os.path.join(image_dir_prefix, 'annotations_ctai.txt'), 'r') as file:
    for line in file:
        parts = line.strip().split(" ")
        img_path = os.path.join(image_dir_prefix, parts[1])
        name = parts[3]
        age_str = parts[5]
        gender = parts[9]

        try:
            age = float(age_str)
        except ValueError:
            age = np.nan

        all_data[name].append({
            'image_path': img_path,
            'age': age,
            'gender': gender
        })

# Impute missing gender values using majority gender per name
name_gender = {}
valid_names_list = []

for name, entries in all_data.items():
    gender_values = [entry['gender'] for entry in entries if entry['gender'].lower() in ['male', 'female']]
    if gender_values:
        most_common_gender = collections.Counter(gender_values).most_common(1)[0][0]
        name_gender[name] = most_common_gender
        valid_names_list.append(name)

# Filter chimpanzees with <10 samples or invalid gender
excludenames = [name for name, entries in all_data.items() 
                if len(entries) < 10 or name not in name_gender]

# Prepare filtered data
imagedirs_filtered, names_filtered, genders_filtered = [], [], []

for name in valid_names_list:
    if name not in excludenames:
        gender = name_gender[name]
        for entry in all_data[name]:
            imagedirs_filtered.append(entry['image_path'])
            names_filtered.append(name)
            genders_filtered.append(gender)

# Encode gender to binary
gender_enc = LabelEncoder()
gender_ids_filtered = gender_enc.fit_transform(genders_filtered)  # Male:1, Female:0 (or vice versa)

# Check distribution
print("Gender distribution after filtering:")
print(collections.Counter(genders_filtered))

# -------------------------
# 2. Train / Val / Test Split
# -------------------------

X_train, X_test, y_train_gender, y_test_gender = train_test_split(
    imagedirs_filtered, gender_ids_filtered, test_size=0.3, stratify=gender_ids_filtered, random_state=42)

X_train, X_val, y_train_gender, y_val_gender = train_test_split(
    X_train, y_train_gender, test_size=0.3, stratify=y_train_gender, random_state=42)

# ----------------------------
# 3. Define Dataset & Loader
# ----------------------------

class CTaiGenderDataset(Dataset):
    def __init__(self, image_paths, gender_labels, transform):
        self.image_paths = image_paths
        self.gender_labels = gender_labels
        self.transform = transform

    def __len__(self):
        return len(self.gender_labels)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        return {'image': image, 'gender': self.gender_labels[idx]}

# Data Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_set = CTaiGenderDataset(X_train, y_train_gender, transform=train_transform)
val_set = CTaiGenderDataset(X_val, y_val_gender, transform=val_test_transform)
test_set = CTaiGenderDataset(X_test, y_test_gender, transform=val_test_transform)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=2)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)

# -----------------------
# 4. Define Gender Model
# -----------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification
model = model.to(device)

# Class weights for imbalance
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(gender_ids_filtered),
    y=gender_ids_filtered
)

weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# -----------------------
# 5. Train the Model
# -----------------------

def train_gender_model(epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0
        for batch in train_loader:
            inputs = batch['image'].float().to(device)
            labels = batch['gender'].detach().clone().long().to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        val_loss, val_acc = evaluate_gender_model()
        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {running_loss/len(train_loader):.4f} "
              f"Train Acc: {correct/total:.4f} Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

# -----------------------
# 6. Evaluate on Validation
# -----------------------

def evaluate_gender_model():
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['image'].float().to(device)
            labels = batch['gender'].detach().clone().long().to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return val_loss / len(val_loader), correct / total

# -----------------------
# 7. Final Testing
# -----------------------

def test_gender_model():
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['image'].float().to(device)
            labels = batch['gender'].detach().clone().long().to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    print(f"\nTest Accuracy: {correct/total:.4f}")
    return np.array(all_preds), np.array(all_labels)

# -----------------------
# 8. Confusion Matrix & Report
# -----------------------

def plot_gender_confusion_matrix(preds, labels):
    cm = confusion_matrix(labels, preds)
    gender_labels_str = gender_enc.classes_
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gender_labels_str)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Gender Classification Confusion Matrix")
    plt.show()

    print("\nClassification Report:\n")
    print(classification_report(labels, preds, target_names=gender_labels_str))

# -----------------------
# 9. Run Training & Testing
# -----------------------

if __name__ == '__main__':
    train_gender_model(epochs=10)
    preds, labels = test_gender_model()
    plot_gender_confusion_matrix(preds, labels)
