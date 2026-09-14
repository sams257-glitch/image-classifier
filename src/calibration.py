import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from dataset import test_loader
from model import create_model


MODEL_PATH = "models/resnet18_baseline.pth"

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = create_model(pretrained=False).to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
    )
)

model.eval()


all_confidences = []
all_predictions = []
all_labels = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        confidences, predictions = torch.max(
            probabilities,
            dim=1,
        )

        all_confidences.extend(
            confidences.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


all_confidences = np.array(all_confidences)
all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)


accuracy = accuracy_score(
    all_labels,
    all_predictions,
)

print(f"Overall accuracy: {accuracy * 100:.2f}%")


bins = np.arange(0.0, 1.01, 0.1)

print("\nConfidence bins:")
print("------------------------------------------")

ece = 0.0
total_samples = len(all_labels)
bin_confidences = []
bin_accuracies = []

for lower, upper in zip(bins[:-1], bins[1:]):

    mask = (
        (all_confidences >= lower)
        & (all_confidences < upper)
    )

    if mask.sum() == 0:
        continue

    bin_accuracy = (
        all_predictions[mask]
        == all_labels[mask]
    ).mean()

    bin_confidence = all_confidences[mask].mean()
    bin_confidences.append(bin_confidence)
    bin_accuracies.append(bin_accuracy)
    

    bin_size = mask.sum()

    calibration_error = abs(
        bin_accuracy - bin_confidence
    )

    weighted_error = (
        bin_size / total_samples
    ) * calibration_error

    ece += weighted_error

    print(
        f"{lower:.1f}-{upper:.1f} | "
        f"Count: {bin_size:4d} | "
        f"Confidence: {bin_confidence * 100:6.2f}% | "
        f"Accuracy: {bin_accuracy * 100:6.2f}%"
    )

print("------------------------------------------")
print(f"Expected Calibration Error (ECE): {ece * 100:.2f}%")
plt.figure(figsize=(7, 7))

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration",
)

plt.plot(
    bin_confidences,
    bin_accuracies,
    marker="o",
    label="ResNet-18",
)

plt.xlabel("Mean confidence")
plt.ylabel("Accuracy")

plt.title("Reliability Diagram")

plt.legend()
plt.grid(True)

plt.savefig(
    "results/reliability_diagram.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()