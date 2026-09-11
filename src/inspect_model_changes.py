import torch
import matplotlib.pyplot as plt
from PIL import Image

from dataset import test_loader, train_dataset, test_dataset
from model import create_model


device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

class_names = train_dataset.classes


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


# Store interesting changed predictions
sea_to_mountain = []
glacier_to_mountain = []
mountain_to_glacier = []
mountain_to_sea = []


with torch.no_grad():
    for batch_index, (images, labels) in enumerate(test_loader):

        images = images.to(device)
        labels = labels.to(device)

        baseline_outputs = baseline_model(images)
        augmented_outputs = augmented_model(images)

        baseline_predictions = baseline_outputs.argmax(dim=1)
        augmented_predictions = augmented_outputs.argmax(dim=1)

        for i in range(len(labels)):

            true_class = class_names[labels[i].item()]

            if true_class != "mountain":
                continue

            baseline = class_names[baseline_predictions[i].item()]
            augmented = class_names[augmented_predictions[i].item()]

            if baseline == augmented:
                continue

            image_index = batch_index * test_loader.batch_size + i

            item = {
                "index": image_index,
                "baseline": baseline,
                "augmented": augmented,
            }

            if baseline == "sea" and augmented == "mountain":
                sea_to_mountain.append(item)

            elif baseline == "glacier" and augmented == "mountain":
                glacier_to_mountain.append(item)

            elif baseline == "mountain" and augmented == "glacier":
                mountain_to_glacier.append(item)

            elif baseline == "mountain" and augmented == "sea":
                mountain_to_sea.append(item)


print("Sea → Mountain:", len(sea_to_mountain))
print("Glacier → Mountain:", len(glacier_to_mountain))
print("Mountain → Glacier:", len(mountain_to_glacier))
print("Mountain → Sea:", len(mountain_to_sea))


# Select a few examples from each important transition
examples = (
    sea_to_mountain[:4]
    + glacier_to_mountain[:4]
    + mountain_to_glacier[:4]
    + mountain_to_sea[:4]
)


fig, axes = plt.subplots(4, 4, figsize=(14, 14))

for ax, item in zip(axes.flat, examples):

    image_path = test_dataset.samples[item["index"]][0]
    image = Image.open(image_path).convert("RGB")

    ax.imshow(image)

    ax.set_title(
        f"Index {item['index']}\n"
        f"Baseline: {item['baseline']}\n"
        f"Augmented: {item['augmented']}"
    )

    ax.axis("off")


plt.tight_layout()

plt.savefig(
    "results/mountain_model_changes.png",
    dpi=150,
)

plt.show()