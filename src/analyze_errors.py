import torch
import matplotlib.pyplot as plt

from dataset import test_loader, train_dataset
from model import create_model


device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using device:", device)

model = create_model().to(device)

model.load_state_dict(
    torch.load(
        "models/resnet18_baseline.pth",
        map_location=device,
    )
)

model.eval()

class_names = train_dataset.classes

wrong_predictions = []

image_index = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        predictions = outputs.argmax(dim=1)

        for i in range(len(labels)):
            if predictions[i] != labels[i]:

                confidence = probabilities[i][predictions[i]].item()

                wrong_predictions.append({
                    "index": image_index + i,
                    "true": labels[i].item(),
                    "predicted": predictions[i].item(),
                    "confidence": confidence,
                })

        image_index += len(labels)


print("Total incorrect predictions:", len(wrong_predictions))


# Sort errors from highest confidence to lowest confidence
wrong_predictions.sort(
    key=lambda x: x["confidence"],
    reverse=True
)

print("\nTop 10 most confident wrong predictions:")

for error in wrong_predictions[:10]:
    print(
        f"True: {class_names[error['true']]:10s} | "
        f"Predicted: {class_names[error['predicted']]:10s} | "
        f"Confidence: {error['confidence']:.2%} | "
        f"Index: {error['index']}"
    )