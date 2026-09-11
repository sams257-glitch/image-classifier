import torch
from dataset import test_loader, train_dataset
from model import create_model


device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using device:", device)

# Load baseline model
baseline_model = create_model().to(device)

baseline_model.load_state_dict(
    torch.load(
        "models/resnet18_baseline.pth",
        map_location=device,
    )
)

baseline_model.eval()


# Load augmented model
augmented_model = create_model().to(device)

augmented_model.load_state_dict(
    torch.load(
        "models/resnet18_augmented.pth",
        map_location=device,
    )
)

augmented_model.eval()


class_names = train_dataset.classes

changed_predictions = []

with torch.no_grad():
    for batch_index, (images, labels) in enumerate(test_loader):

        images = images.to(device)
        labels = labels.to(device)

        baseline_outputs = baseline_model(images)
        augmented_outputs = augmented_model(images)

        baseline_predictions = baseline_outputs.argmax(dim=1)
        augmented_predictions = augmented_outputs.argmax(dim=1)

        for i in range(len(labels)):

            true_class = labels[i].item()
            baseline_prediction = baseline_predictions[i].item()
            augmented_prediction = augmented_predictions[i].item()

            # Focus only on true mountain images
            if class_names[true_class] != "mountain":
                continue

            if baseline_prediction != augmented_prediction:
                image_index = batch_index * test_loader.batch_size + i

                changed_predictions.append(
                    {
                        "index": image_index,
                        "baseline": class_names[baseline_prediction],
                        "augmented": class_names[augmented_prediction],
                    }
                )


print("\nChanged predictions for mountain images:")
print("Total:", len(changed_predictions))

for item in changed_predictions:
    print(
        f"Index: {item['index']} | "
        f"Baseline: {item['baseline']} | "
        f"Augmented: {item['augmented']}"
    )
corrected = 0
broken = 0

transition_counts = {}


for item in changed_predictions:

    baseline = item["baseline"]
    augmented = item["augmented"]

    transition = f"{baseline} -> {augmented}"

    transition_counts[transition] = (
        transition_counts.get(transition, 0) + 1
    )

    baseline_correct = baseline == "mountain"
    augmented_correct = augmented == "mountain"

    if not baseline_correct and augmented_correct:
        corrected += 1

    elif baseline_correct and not augmented_correct:
        broken += 1


print("\nPrediction transitions:")

for transition, count in sorted(
    transition_counts.items(),
    key=lambda x: x[1],
    reverse=True,
):
    print(f"{transition}: {count}")


print("\nMountain corrections:", corrected)
print("Mountain predictions made worse:", broken)