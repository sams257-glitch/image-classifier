import torch
from PIL import Image, ImageFilter
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

from dataset import test_dataset
from model import create_model


# --------------------------------------------------
# 1. Load model
# --------------------------------------------------

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


# --------------------------------------------------
# 2. Preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# --------------------------------------------------
# 3. Find mountain images
# --------------------------------------------------

mountain_class = test_dataset.classes.index("mountain")

correct_indices = []
wrong_indices = []


with torch.no_grad():

    for index in range(len(test_dataset)):

        image, label = test_dataset[index]

        if label != mountain_class:
            continue

        image_tensor = image.unsqueeze(0).to(device)

        output = model(image_tensor)

        prediction = output.argmax(dim=1).item()

        if prediction == mountain_class:
            correct_indices.append(index)
        else:
            wrong_indices.append(index)


# Balance the groups

sample_size = min(
    len(correct_indices),
    len(wrong_indices)
)

correct_indices = correct_indices[:sample_size]
wrong_indices = wrong_indices[:sample_size]

print()
print("Correctly classified:", len(correct_indices))
print("Misclassified:", len(wrong_indices))


# --------------------------------------------------
# 4. Convert tensor back to PIL
# --------------------------------------------------

def tensor_to_pil(image):

    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    image = image * std + mean
    image = torch.clamp(image, 0, 1)

    return transforms.ToPILImage()(image)


# --------------------------------------------------
# 5. Get mountain probability
# --------------------------------------------------

def get_mountain_probability(image):

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

    return probabilities[0, mountain_class].item()


# --------------------------------------------------
# 6. Create regional perturbations
# --------------------------------------------------

def create_region_versions(image):

    width, height = image.size

    top_blurred = image.copy()
    bottom_blurred = image.copy()

    # Top half

    top_half = image.crop(
        (0, 0, width, height // 2)
    )

    top_half = top_half.filter(
        ImageFilter.GaussianBlur(radius=8)
    )

    top_blurred.paste(
        top_half,
        (0, 0)
    )

    # Bottom half

    bottom_half = image.crop(
        (0, height // 2, width, height)
    )

    bottom_half = bottom_half.filter(
        ImageFilter.GaussianBlur(radius=8)
    )

    bottom_blurred.paste(
        bottom_half,
        (0, height // 2)
    )

    return top_blurred, bottom_blurred


# --------------------------------------------------
# 7. Run experiment
# --------------------------------------------------

def run_experiment(indices):

    top_changes = []
    bottom_changes = []

    for index in indices:

        image, _ = test_dataset[index]

        image = tensor_to_pil(image)

        original_probability = (
            get_mountain_probability(image)
        )

        top_blurred, bottom_blurred = (
            create_region_versions(image)
        )

        top_probability = (
            get_mountain_probability(top_blurred)
        )

        bottom_probability = (
            get_mountain_probability(bottom_blurred)
        )

        top_changes.append(
            original_probability - top_probability
        )

        bottom_changes.append(
            original_probability - bottom_probability
        )

    return top_changes, bottom_changes


# --------------------------------------------------
# 8. Run experiment
# --------------------------------------------------

print()
print("Running experiment...")
print()

correct_top, correct_bottom = run_experiment(
    correct_indices
)

wrong_top, wrong_bottom = run_experiment(
    wrong_indices
)


# --------------------------------------------------
# 9. Statistics
# --------------------------------------------------

def print_statistics(name, values):

    values = np.array(values)

    print(name)

    print(
        f"Mean:   {values.mean():+.2%}"
    )

    print(
        f"Median: {np.median(values):+.2%}"
    )

    print(
        f"Std:    {values.std():.2%}"
    )

    print(
        f"Min:    {values.min():+.2%}"
    )

    print(
        f"Max:    {values.max():+.2%}"
    )

    print()


print("=" * 60)
print("TOP REGION")
print("=" * 60)
print()

print("CORRECTLY CLASSIFIED")
print_statistics(
    "Top blur change:",
    correct_top
)

print("MISCLASSIFIED")
print_statistics(
    "Top blur change:",
    wrong_top
)


print("=" * 60)
print("BOTTOM REGION")
print("=" * 60)
print()

print("CORRECTLY CLASSIFIED")
print_statistics(
    "Bottom blur change:",
    correct_bottom
)

print("MISCLASSIFIED")
print_statistics(
    "Bottom blur change:",
    wrong_bottom
)


# --------------------------------------------------
# 10. Visualize distributions
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.boxplot(
    [
        correct_top,
        wrong_top,
        correct_bottom,
        wrong_bottom
    ],
    tick_labels=[
        "Correct\nTop",
        "Wrong\nTop",
        "Correct\nBottom",
        "Wrong\nBottom"
    ]
)

plt.axhline(
    0,
    linestyle="--"
)

plt.ylabel(
    "Change in mountain probability"
)

plt.title(
    "Effect of Regional Blur on Mountain Probability"
)

plt.tight_layout()

plt.savefig(
    "results/region_probability_distribution.png"
)



# --------------------------------------------------
# 11. Statistical comparison
# --------------------------------------------------

from scipy.stats import mannwhitneyu


def compare_groups(correct, wrong, name):

    statistic, p_value = mannwhitneyu(
        correct,
        wrong,
        alternative="two-sided"
    )

    print("=" * 60)
    print(name)
    print("=" * 60)

    print(
        f"Mann-Whitney U statistic: "
        f"{statistic:.2f}"
    )

    print(
        f"p-value: "
        f"{p_value:.6f}"
    )

    print()


compare_groups(
    correct_top,
    wrong_top,
    "TOP REGION: Correct vs Misclassified"
)

compare_groups(
    correct_bottom,
    wrong_bottom,
    "BOTTOM REGION: Correct vs Misclassified"
)

def rank_biserial_effect(correct, wrong):

    u, _ = mannwhitneyu(
        correct,
        wrong,
        alternative="two-sided"
    )

    n1 = len(correct)
    n2 = len(wrong)

    effect = (
        (2 * u) / (n1 * n2)
    ) - 1

    return effect


top_effect = rank_biserial_effect(
    correct_top,
    wrong_top
)

bottom_effect = rank_biserial_effect(
    correct_bottom,
    wrong_bottom
)


print("=" * 60)
print("EFFECT SIZES")
print("=" * 60)

print(
    f"Top region rank-biserial correlation: "
    f"{top_effect:+.3f}"
)

print(
    f"Bottom region rank-biserial correlation: "
    f"{bottom_effect:+.3f}"
)

plt.show()