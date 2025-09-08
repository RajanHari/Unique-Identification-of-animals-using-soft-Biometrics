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
            age = np.nan # Assign NaN if age is not a number

        all_data[name].append({
            'image_path': img_path,
            'age': age,
            'gender': gender
        })

# Calculate average age per name and filter names with no valid age
name_avg_age = {}
valid_names_list = []
for name, entries in all_data.items():
    valid_ages_for_name = [entry['age'] for entry in entries if not np.isnan(entry['age'])]
    if valid_ages_for_name: # Only keep names with at least one valid age
        name_avg_age[name] = np.nanmean(valid_ages_for_name)
        valid_names_list.append(name)

# Exclude chimpanzees with <10 samples AND no valid age
excludenames = [name for name, entries in all_data.items() if len(entries) < 10 or name not in valid_names_list]

# Prepare filtered data
imagedirs_filtered, names_filtered, ages_filtered, genders_filtered = [], [], [], []

for name in valid_names_list:
    if name not in excludenames:
        avg_age = name_avg_age[name]
        for entry in all_data[name]:
            imagedirs_filtered.append(entry['image_path'])
            names_filtered.append(name) # Use the name for filtering
            ages_filtered.append(avg_age) # Use the average age for all entries of this name
            genders_filtered.append(entry['gender'])


# Define threshold for age grouping and convert to age groups
age_multiple_threshold = 8
age_groups_filtered = [int(age // age_multiple_threshold) for age in ages_filtered]


# Label encode age group classes
age_enc = LabelEncoder()
agegroup_ids_filtered = age_enc.fit_transform(age_groups_filtered)


# To check class distributions across age groups
print("Age group distribution after filtering:")
print(collections.Counter(age_groups_filtered))

# -------------------------
# 2. Train / Val / Test Split
# -------------------------

# Using age_groups_filtered for stratification
X_train, X_test, y_train_age, y_test_age = train_test_split(
    imagedirs_filtered, agegroup_ids_filtered, test_size=0.3, stratify=agegroup_ids_filtered, random_state=42)

X_train, X_val, y_train_age, y_val_age = train_test_split(
    X_train, y_train_age, test_size=0.3, stratify=y_train_age, random_state=42)

# ----------------------------
# 3. Define Dataset & Loader
# ----------------------------

class CTaiAgeDataset(Dataset):
    def __init__(self, image_paths, age_labels, transform):
        self.image_paths = image_paths
        self.age_labels = age_labels
        self.transform = transform

    def __len__(self):
        return len(self.age_labels)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        return {'image': image, 'age': self.age_labels[idx]}




train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # --- Non-geometric augmentations ---
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.2),

    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Validation and Test should NOT have augmentations
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


train_set = CTaiAgeDataset(X_train, y_train_age, transform=train_transform)
val_set = CTaiAgeDataset(X_val, y_val_age, transform=val_test_transform)
test_set = CTaiAgeDataset(X_test, y_test_age, transform=val_test_transform)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=2)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)

# -----------------------
# 4. Define Age Model
# -----------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_age_classes = len(age_enc.classes_)

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_age_classes)
model = model.to(device)



class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(agegroup_ids_filtered), # Use agegroup_ids_filtered
    y=age_groups_filtered # Use age_groups_filtered
)

weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# -----------------------
# 5. Train the Model
# -----------------------

def train_age_model(epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0
        for batch in train_loader:
            inputs = batch['image'].float().to(device)
            # Use detach().clone() to create a new tensor from batch['age']
            labels = batch['age'].detach().clone().long().to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        val_loss, val_acc = evaluate_age_model()
        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {running_loss/len(train_loader):.4f} "
              f"Train Acc: {correct/total:.4f} Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

# -----------------------
# 6. Evaluate on Validation
# -----------------------

def evaluate_age_model():
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['image'].float().to(device)
            # Use detach().clone() to create a new tensor from batch['age']
            labels = batch['age'].detach().clone().long().to(device)
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

def test_age_model():
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['image'].float().to(device)
            # Use detach().clone() to create a new tensor from batch['age']
            labels = batch['age'].detach().clone().long().to(device)
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

def plot_age_confusion_matrix(preds, labels):
    cm = confusion_matrix(labels, preds)
    # Convert age_enc.classes_ to strings
    age_group_labels_str = [str(label) for label in age_enc.classes_]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=age_group_labels_str)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Age Group Confusion Matrix")
    plt.show()

    print("\nClassification Report:\n")
    print(classification_report(labels, preds, target_names=age_group_labels_str))

# -----------------------
# 9. Run Training & Testing
# -----------------------

train_age_model(epochs=10)
preds, labels = test_age_model()
plot_age_confusion_matrix(preds, labels)