import torch
from collections import Counter

from dataset import test_loader, train_dataset
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

class_names = train_dataset.classes

error_pairs = Counter()

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        for true, predicted in zip(labels, predictions):

            true_class = true.item()
            predicted_class = predicted.item()

            if true_class != predicted_class:
                error_pairs[
                    (
                        class_names[true_class],
                        class_names[predicted_class],
                    )
                ] += 1


print("Most common classification errors:\n")

for (true_class, predicted_class), count in error_pairs.most_common():

    print(
        f"{true_class:10s} → "
        f"{predicted_class:10s}: "
        f"{count}"
    )