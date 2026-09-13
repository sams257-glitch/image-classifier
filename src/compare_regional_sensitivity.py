import torch
import numpy as np
from PIL import ImageFilter

from dataset import test_dataset
from model import create_model
from scipy.stats import wilcoxon


device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

CLASS_NAMES = test_dataset.classes
MOUNTAIN_INDEX = CLASS_NAMES.index("mountain")


def load_model(path):
    model = create_model().to(device)

    model.load_state_dict(
        torch.load(
            path,
            map_location=device,
        )
    )

    model.eval()
    return model


baseline_model = load_model(
    "models/resnet18_baseline.pth"
)

augmented_model = load_model(
    "models/resnet18_augmented.pth"
)


def blur_region(image, region):
    """
    Blur either the top or bottom half of an image.
    """
    image = image.copy()

    width, height = image.size

    if region == "top":
        box = (0, 0, width, height // 2)
    else:
        box = (0, height // 2, width, height)

    crop = image.crop(box)
    crop = crop.filter(ImageFilter.GaussianBlur(radius=8))
    image.paste(crop, box)

    return image


def predict(model, image):
    tensor = test_dataset.transform(image)
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)

    return probabilities[0, MOUNTAIN_INDEX].item()


# Find mountain images.
mountain_indices = [
    i
    for i, (_, label) in enumerate(test_dataset.samples)
    if label == MOUNTAIN_INDEX
]

print("Device:", device)
print("Mountain images:", len(mountain_indices))


results = {
    "baseline": {
        "top": [],
        "bottom": [],
    },
    "augmented": {
        "top": [],
        "bottom": [],
    },
}


for index in mountain_indices:

    image_path, label = test_dataset.samples[index]

    from PIL import Image
    image = Image.open(image_path).convert("RGB")

    for region in ["top", "bottom"]:

        blurred_image = blur_region(
            image,
            region,
        )

        original_baseline = predict(
            baseline_model,
            image,
        )

        blurred_baseline = predict(
            baseline_model,
            blurred_image,
        )

        original_augmented = predict(
            augmented_model,
            image,
        )

        blurred_augmented = predict(
            augmented_model,
            blurred_image,
        )

        baseline_change = (
            original_baseline - blurred_baseline
        )

        augmented_change = (
            original_augmented - blurred_augmented
        )

        results["baseline"][region].append(
            baseline_change
        )

        results["augmented"][region].append(
            augmented_change
        )


for model_name in ["baseline", "augmented"]:

    print(f"\n{model_name.upper()}")

    for region in ["top", "bottom"]:

        values = np.array(
            results[model_name][region]
        )

        print(
            f"{region.capitalize()} region:"
        )

        print(
            f"Mean change: {values.mean() * 100:.2f}%"
        )

        print(
            f"Median change: {np.median(values) * 100:.2f}%"
        )

        print(
            f"Std: {values.std() * 100:.2f}%"
        )

print("\nSTATISTICAL COMPARISON")

for region in ["top", "bottom"]:
    baseline = np.array(results["baseline"][region])
    augmented = np.array(results["augmented"][region])

    differences = augmented - baseline

    statistic, p_value = wilcoxon(
        baseline,
        augmented,
    )

    n = len(differences)

    # Remove zero differences because they contribute no rank.
    nonzero = differences[differences != 0]
    n_nonzero = len(nonzero)

    # Expected value and standard deviation
    # of the Wilcoxon signed-rank statistic.
    mean_w = n_nonzero * (n_nonzero + 1) / 4

    std_w = np.sqrt(
        n_nonzero
        * (n_nonzero + 1)
        * (2 * n_nonzero + 1)
        / 24
    )

    # Convert Wilcoxon statistic to z-score.
    z = (statistic - mean_w) / std_w

    # Effect size r.
    effect_size = abs(z) / np.sqrt(n_nonzero)

    print(f"\n{region.capitalize()} region:")
    print(f"Wilcoxon statistic: {statistic:.2f}")
    print(f"p-value: {p_value:.6f}")
    print(f"Mean paired difference: {differences.mean() * 100:.2f}%")
    print(f"Median paired difference: {np.median(differences) * 100:.2f}%")
    print(f"Z-score: {z:.3f}")
    print(f"Effect size r: {effect_size:.3f}")