import os
import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Avoid duplicate library errors on some systems
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
image_size = 224
batch_size = 64
num_workers = 0
learning_rate = 1e-4
num_epochs = 10
class_names = ['COVID', 'Normal', 'Pneumonia', 'Pneumothorax', 'Tuberculosis']

# Define image transformations for training and testing
train_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),
    transforms.Resize((image_size, image_size)),
    transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load dataset and apply transformations
dir = r'C:\Users\HP\Downloads\c320\320\Lung-Disease-Detection-using-CNN\test'
full_dataset = datasets.ImageFolder(dir, transform=train_transform)

# Split dataset into train, validation, and test
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
test_dataset.dataset.transform = test_transform

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Load pretrained ResNet101 model and customize the final layer
model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.fc.in_features, 5)
)
model = model.to(device)

# Loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

# Freeze pretrained layers, only train final layer
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

# Lists to store training/validation metrics
train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []

# Federated training simulation: each batch acts like a separate client
for epoch in range(num_epochs):
    print(f"Starting epoch {epoch + 1}")
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    client_weights = []

    for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        # Clone model to simulate a client
        local_model = models.resnet101(weights=None)
        local_model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(local_model.fc.in_features, 5)
        )
        local_model.load_state_dict(model.state_dict())
        local_model = local_model.to(device)
        local_model.train()

        # Local optimization step
        opt = optim.Adam(local_model.parameters(), lr=learning_rate)
        opt.zero_grad()
        loss = criterion(local_model(inputs), labels)
        loss.backward()
        opt.step()

        # Save client model weights
        client_weights.append(local_model.state_dict())

        # Track metrics
        with torch.no_grad():
            outputs = local_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

    # Federated averaging of model weights
    avg_weights = model.state_dict()
    for k in avg_weights.keys():
        avg_weights[k] = torch.stack([cw[k] for cw in client_weights], 0).mean(0)
    model.load_state_dict(avg_weights)

    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(100 * correct / total)
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Average Loss: {train_losses[-1]:.4f}, Accuracy: {train_accuracies[-1]:.2f}%")

    # Validation step
    model.eval()
    running_val_loss, correct_val, total_val = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss = criterion(outputs, labels)
            running_val_loss += val_loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_losses.append(running_val_loss / len(val_loader))
    val_accuracies.append(100 * correct_val / total_val)
    print(f"Validation - Loss: {val_losses[-1]:.4f}, Accuracy: {val_accuracies[-1]:.2f}%")
    scheduler.step(val_losses[-1])

# Final evaluation on test set
model.eval()
running_test_loss, correct_test, total_test = 0.0, 0, 0
y_true, y_pred = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

average_test_loss = running_test_loss / len(test_loader)
test_accuracy = 100 * correct_test / total_test
print(f"\nTest Loss: {average_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

# Plot training/validation metrics and confusion matrix
plt.figure(figsize=(18, 5))
epochs = list(range(1, num_epochs + 1))

# Plot losses
plt.subplot(1, 3, 1)
plt.plot(epochs, train_losses, label='Train Loss', marker='o')
plt.plot(epochs, val_losses, label='Val Loss', marker='o')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot accuracies
plt.subplot(1, 3, 2)
plt.plot(epochs, train_accuracies, label='Train Acc', marker='o')
plt.plot(epochs, val_accuracies, label='Val Acc', marker='o')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

# Plot confusion matrix
unique_labels = np.unique(y_true + y_pred)
if len(unique_labels) != len(class_names):
    class_names = [f"Class {i}" for i in unique_labels]
cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.xticks(rotation=45)
plt.show()

# Save the trained model
torch.save(model.state_dict(), 'resnet101_lung_model.pth')
