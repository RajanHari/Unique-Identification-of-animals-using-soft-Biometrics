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
import torch.nn.functional as F

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

# Calculate average age per name and impute gender
name_avg_age = {}
name_gender = {}
valid_names_list = []

for name, entries in all_data.items():
    # Handle age
    valid_ages_for_name = [entry['age'] for entry in entries if not np.isnan(entry['age'])]
    if valid_ages_for_name:
        name_avg_age[name] = np.nanmean(valid_ages_for_name)
    
    # Handle gender
    gender_values = [entry['gender'] for entry in entries if entry['gender'].lower() in ['male', 'female']]
    if gender_values:
        most_common_gender = collections.Counter(gender_values).most_common(1)[0][0]
        name_gender[name] = most_common_gender
    
    # Only keep names with both valid age and gender
    if name in name_avg_age and name in name_gender:
        valid_names_list.append(name)

# Exclude chimpanzees with <10 samples or missing age/gender
excludenames = [name for name, entries in all_data.items() 
                if len(entries) < 10 or name not in valid_names_list]

# Prepare filtered data
imagedirs_filtered, names_filtered, ages_filtered, genders_filtered = [], [], [], []

for name in valid_names_list:
    if name not in excludenames:
        avg_age = name_avg_age[name]
        gender = name_gender[name]
        for entry in all_data[name]:
            imagedirs_filtered.append(entry['image_path'])
            names_filtered.append(name)
            ages_filtered.append(avg_age)
            genders_filtered.append(gender)

# Create age groups and encode labels
age_multiple_threshold = 8
age_groups_filtered = [int(age // age_multiple_threshold) for age in ages_filtered]

# Encode all labels
name_enc = LabelEncoder()
age_enc = LabelEncoder()
gender_enc = LabelEncoder()

name_ids_filtered = name_enc.fit_transform(names_filtered)
agegroup_ids_filtered = age_enc.fit_transform(age_groups_filtered)
gender_ids_filtered = gender_enc.fit_transform(genders_filtered)

print(f"Number of unique names: {len(name_enc.classes_)}")
print(f"Number of age groups: {len(age_enc.classes_)}")
print(f"Number of genders: {len(gender_enc.classes_)}")

# -------------------------
# 2. Train / Val / Test Split
# -------------------------

# Stratify by name for balanced split
X_train, X_test, y_train_name, y_test_name, y_train_age, y_test_age, y_train_gender, y_test_gender = train_test_split(
    imagedirs_filtered, name_ids_filtered, agegroup_ids_filtered, gender_ids_filtered,
    test_size=0.3, stratify=name_ids_filtered, random_state=42)

X_train, X_val, y_train_name, y_val_name, y_train_age, y_val_age, y_train_gender, y_val_gender = train_test_split(
    X_train, y_train_name, y_train_age, y_train_gender,
    test_size=0.3, stratify=y_train_name, random_state=42)

# ----------------------------
# 3. Define Multi-task Dataset
# ----------------------------

class CTaiMultiTaskDataset(Dataset):
    def __init__(self, image_paths, name_labels, age_labels, gender_labels, transform):
        self.image_paths = image_paths
        self.name_labels = name_labels
        self.age_labels = age_labels
        self.gender_labels = gender_labels
        self.transform = transform

    def __len__(self):
        return len(self.name_labels)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        return {
            'image': image,
            'name': self.name_labels[idx],
            'age': self.age_labels[idx],
            'gender': self.gender_labels[idx]
        }

# Data Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_set = CTaiMultiTaskDataset(X_train, y_train_name, y_train_age, y_train_gender, transform=train_transform)
val_set = CTaiMultiTaskDataset(X_val, y_val_name, y_val_age, y_val_gender, transform=val_test_transform)
test_set = CTaiMultiTaskDataset(X_test, y_test_name, y_test_age, y_test_gender, transform=val_test_transform)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=2)

# ----------------------------
# 4. Individual Models
# ----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Age Model
class AgeModel(nn.Module):
    def __init__(self, num_classes):
        super(AgeModel, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

# Gender Model
class GenderModel(nn.Module):
    def __init__(self, num_classes=2):
        super(GenderModel, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

# Name Model (baseline)
class NameModel(nn.Module):
    def __init__(self, num_classes):
        super(NameModel, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

# ----------------------------
# 5. Ensemble Model
# ----------------------------

class EnsembleNameClassifier(nn.Module):
    def __init__(self, num_names, num_ages, num_genders=2):
        super(EnsembleNameClassifier, self).__init__()
        
        # Individual models
        self.age_model = AgeModel(num_ages)
        self.gender_model = GenderModel(num_genders)
        self.name_model = NameModel(num_names)
        
        # Feature extractor for ensemble
        self.feature_extractor = models.resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Identity()  # Remove final layer
        
        # Meta classifier
        # Input: ResNet features (512) + age predictions (num_ages) + gender predictions (2)
        ensemble_input_size = 512 + num_ages + num_genders
        self.meta_classifier = nn.Sequential(
            nn.Linear(ensemble_input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_names)
        )
    
    def forward(self, x, return_individual=False):
        # Get predictions from individual models
        age_logits = self.age_model(x)
        gender_logits = self.gender_model(x)
        name_logits = self.name_model(x)
        
        # Get features for meta classifier
        features = self.feature_extractor(x)
        age_probs = F.softmax(age_logits, dim=1)
        gender_probs = F.softmax(gender_logits, dim=1)
        
        # Concatenate features
        ensemble_features = torch.cat([features, age_probs, gender_probs], dim=1)
        
        # Meta classifier prediction
        ensemble_logits = self.meta_classifier(ensemble_features)
        
        if return_individual:
            return {
                'ensemble': ensemble_logits,
                'name': name_logits,
                'age': age_logits,
                'gender': gender_logits
            }
        
        return ensemble_logits

# Initialize models
num_names = len(name_enc.classes_)
num_ages = len(age_enc.classes_)
num_genders = len(gender_enc.classes_)

ensemble_model = EnsembleNameClassifier(num_names, num_ages, num_genders).to(device)

# ----------------------------
# 6. Training Functions
# ----------------------------



def train_individual_models(epochs=10):
    """Train individual age and gender models first"""
    
    # Train age model
    print("Training Age Model...")
    age_criterion = nn.CrossEntropyLoss()
    age_optimizer = optim.Adam(ensemble_model.age_model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        ensemble_model.age_model.train()
        running_loss, correct, total = 0, 0, 0
        
        for batch in train_loader:
            inputs = batch['image'].float().to(device)
            labels = batch['age'].long().to(device)
            
            age_optimizer.zero_grad()
            outputs = ensemble_model.age_model(inputs)
            loss = age_criterion(outputs, labels)
            loss.backward()
            age_optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        if epoch % 2 == 0:
            print(f"Age Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f} "
                  f"Acc: {correct/total:.4f}")
    
    # Train gender model
    print("\nTraining Gender Model...")
    gender_criterion = nn.CrossEntropyLoss()
    gender_optimizer = optim.Adam(ensemble_model.gender_model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        ensemble_model.gender_model.train()
        running_loss, correct, total = 0, 0, 0
        
        for batch in train_loader:
            inputs = batch['image'].float().to(device)
            labels = batch['gender'].long().to(device)
            
            gender_optimizer.zero_grad()
            outputs = ensemble_model.gender_model(inputs)
            loss = gender_criterion(outputs, labels)
            loss.backward()
            gender_optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        if epoch % 2 == 0:
            print(f"Gender Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f} "
                  f"Acc: {correct/total:.4f}")

def train_ensemble_model(epochs=15):
    """Train the ensemble model end-to-end"""
    
    # Freeze individual models initially (optional)
    for param in ensemble_model.age_model.parameters():
        param.requires_grad = False
    for param in ensemble_model.gender_model.parameters():
        param.requires_grad = False
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': ensemble_model.name_model.parameters()},
        {'params': ensemble_model.feature_extractor.parameters()},
        {'params': ensemble_model.meta_classifier.parameters()}
    ], lr=1e-3)
    
    print("\nTraining Ensemble Model...")
    
    for epoch in range(epochs):
        ensemble_model.train()
        running_loss, correct, total = 0, 0, 0
        
        for batch in train_loader:
            inputs = batch['image'].float().to(device)
            name_labels = batch['name'].long().to(device)
            
            optimizer.zero_grad()
            ensemble_outputs = ensemble_model(inputs)
            loss = criterion(ensemble_outputs, name_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = ensemble_outputs.max(1)
            total += name_labels.size(0)
            correct += predicted.eq(name_labels).sum().item()
        
        # Validation
        val_acc = evaluate_ensemble_model()
        print(f"Ensemble Epoch [{epoch+1}/{epochs}] Train Loss: {running_loss/len(train_loader):.4f} "
              f"Train Acc: {correct/total:.4f} Val Acc: {val_acc:.4f}")
        
        # Unfreeze individual models after a few epochs for fine-tuning
        if epoch == 5:
            for param in ensemble_model.age_model.parameters():
                param.requires_grad = True
            for param in ensemble_model.gender_model.parameters():
                param.requires_grad = True
            
            # Add individual model parameters to optimizer
            optimizer.add_param_group({'params': ensemble_model.age_model.parameters(), 'lr': 1e-4})
            optimizer.add_param_group({'params': ensemble_model.gender_model.parameters(), 'lr': 1e-4})

def evaluate_ensemble_model():
    """Evaluate ensemble model on validation set"""
    ensemble_model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['image'].float().to(device)
            name_labels = batch['name'].long().to(device)
            
            outputs = ensemble_model(inputs)
            _, predicted = outputs.max(1)
            total += name_labels.size(0)
            correct += predicted.eq(name_labels).sum().item()
    
    return correct / total

def test_models():
    """Test both baseline name model and ensemble model"""
    ensemble_model.eval()
    
    ensemble_correct, baseline_correct, total = 0, 0, 0
    ensemble_preds, baseline_preds, true_labels = [], [], []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['image'].float().to(device)
            name_labels = batch['name'].long().to(device)
            
            # Get predictions from both models
            outputs = ensemble_model(inputs, return_individual=True)
            ensemble_pred = outputs['ensemble'].max(1)[1]
            baseline_pred = outputs['name'].max(1)[1]
            
            # Collect predictions
            ensemble_preds.extend(ensemble_pred.cpu().numpy())
            baseline_preds.extend(baseline_pred.cpu().numpy())
            true_labels.extend(name_labels.cpu().numpy())
            
            # Calculate accuracies
            total += name_labels.size(0)
            ensemble_correct += ensemble_pred.eq(name_labels).sum().item()
            baseline_correct += baseline_pred.eq(name_labels).sum().item()
    
    ensemble_acc = ensemble_correct / total
    baseline_acc = baseline_correct / total
    
    print(f"\n=== TEST RESULTS ===")
    print(f"Baseline Name Model Accuracy: {baseline_acc:.4f}")
    print(f"Ensemble Model Accuracy: {ensemble_acc:.4f}")
    print(f"Improvement: {ensemble_acc - baseline_acc:.4f} ({((ensemble_acc/baseline_acc - 1) * 100):.2f}%)")
    
    return np.array(ensemble_preds), np.array(baseline_preds), np.array(true_labels)

def plot_confusion_matrices(ensemble_preds, baseline_preds, true_labels):
    """Plot confusion matrices for both models"""
    
    # Select top 20 most frequent names for visualization
    name_counts = collections.Counter(true_labels)
    top_names = [name for name, _ in name_counts.most_common(20)]
    top_name_indices = [i for i, label in enumerate(true_labels) if label in top_names]
    
    if len(top_name_indices) > 0:
        subset_true = [true_labels[i] for i in top_name_indices]
        subset_ensemble = [ensemble_preds[i] for i in top_name_indices]
        subset_baseline = [baseline_preds[i] for i in top_name_indices]
        
        # Create label mapping for top names
        unique_labels = sorted(list(set(subset_true)))
        label_names = [name_enc.classes_[i] for i in unique_labels]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Baseline confusion matrix
        cm_baseline = confusion_matrix(subset_true, subset_baseline, labels=unique_labels)
        disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_baseline, display_labels=label_names)
        disp1.plot(ax=ax1, cmap='Blues', xticks_rotation=45)
        ax1.set_title("Baseline Name Model (Top 20 Names)")
        
        # Ensemble confusion matrix  
        cm_ensemble = confusion_matrix(subset_true, subset_ensemble, labels=unique_labels)
        disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_ensemble, display_labels=label_names)
        disp2.plot(ax=ax2, cmap='Greens', xticks_rotation=45)
        ax2.set_title("Ensemble Name Model (Top 20 Names)")
        
        plt.tight_layout()
        plt.show()

# ----------------------------
# 7. Run Training & Testing
# ----------------------------

if __name__ == '__main__':
    # Step 1: Train individual models
    train_individual_models(epochs=8)
    
    # Step 2: Train ensemble model
    train_ensemble_model(epochs=100)
    
    # Step 3: Test and compare
    ensemble_preds, baseline_preds, true_labels = test_models()
    
    # Step 4: Visualize results
    plot_confusion_matrices(ensemble_preds, baseline_preds, true_labels)
    
    # Additional analysis
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(imagedirs_filtered)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")