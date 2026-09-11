import torch
import matplotlib.pyplot as plt

from dataset import test_loader, test_dataset
from model import create_model


device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

model = create_model().to(device)

model.load_state_dict(
    torch.load(
        "models/resnet18_baseline.pth",
        map_location=device,
    )
)

model.eval()

class_names = test_dataset.classes

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


wrong_predictions.sort(
    key=lambda x: x["confidence"],
    reverse=True
)


# Show the 10 most confident mistakes
fig, axes = plt.subplots(2, 5, figsize=(15, 7))

for ax, error in zip(axes.flat, wrong_predictions[:10]):

    image, label = test_dataset[error["index"]]

    # Undo ImageNet normalization so the image displays correctly
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    image = image * std + mean
    image = image.clamp(0, 1)

    ax.imshow(image.permute(1, 2, 0))

    ax.set_title(
        f"True: {class_names[error['true']]}\n"
        f"Pred: {class_names[error['predicted']]}\n"
        f"Confidence: {error['confidence']:.1%}"
    )

    ax.axis("off")


plt.tight_layout()
plt.savefig("results/top_confident_errors.png")
plt.show()