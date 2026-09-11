import torch
from PIL import Image, ImageFilter
from torchvision import transforms

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


# Use an equal number from each group

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
# 4. Convert dataset tensor back to PIL
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
# 5. Get probability of mountain
# --------------------------------------------------

def get_true_class_probability(image):

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

    return probabilities[0, mountain_class].item()


# --------------------------------------------------
# 6. Blur top/bottom
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

        # Original probability of "mountain"

        original_probability = (
            get_true_class_probability(image)
        )

        # Create perturbations

        top_blurred, bottom_blurred = (
            create_region_versions(image)
        )

        # Probabilities after perturbation

        top_probability = (
            get_true_class_probability(top_blurred)
        )

        bottom_probability = (
            get_true_class_probability(bottom_blurred)
        )

        # Change in probability

        top_change = (
            original_probability - top_probability
        )

        bottom_change = (
            original_probability - bottom_probability
        )

        top_changes.append(top_change)
        bottom_changes.append(bottom_change)

    return top_changes, bottom_changes


# --------------------------------------------------
# 8. Run both groups
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
# 9. Average changes
# --------------------------------------------------

def average(values):

    return sum(values) / len(values)


print("=" * 55)
print("TRUE-CLASS PROBABILITY RESULTS")
print("=" * 55)

print()

print("CORRECTLY CLASSIFIED MOUNTAINS")

print(
    f"Top blur change: "
    f"{average(correct_top):+.2%}"
)

print(
    f"Bottom blur change: "
    f"{average(correct_bottom):+.2%}"
)

print()

print("MISCLASSIFIED MOUNTAINS")

print(
    f"Top blur change: "
    f"{average(wrong_top):+.2%}"
)

print(
    f"Bottom blur change: "
    f"{average(wrong_bottom):+.2%}"
)