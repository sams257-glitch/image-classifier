import torch
from dataset import test_loader, train_dataset
from model import create_model

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using device:", device)

model = create_model().to(device)

model.load_state_dict(
    torch.load(
        "models/resnet18_augmented.pth",
        map_location=device,
    )
)

model.eval()

all_labels = []
all_predictions = []

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")
print("Total labels collected:", len(all_labels))
print("Total predictions collected:", len(all_predictions))

from sklearn.metrics import classification_report

class_names = train_dataset.classes

report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names,
)

print("\nClassification Report:")
print(report)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(
    all_labels,
    all_predictions,
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names,
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - ResNet-18 Augmented")

plt.tight_layout()

plt.savefig("results/augmented_confusion_matrix.png")

plt.show()