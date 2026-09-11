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
# 2. Image preprocessing
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


print()
print("Correctly classified mountain images:",
      len(correct_indices))

print("Misclassified mountain images:",
      len(wrong_indices))


# --------------------------------------------------
# 4. Limit the experiment to 50 + 50
# --------------------------------------------------

correct_indices = correct_indices[:50]
wrong_indices = wrong_indices[:50]


# --------------------------------------------------
# 5. Function to get confidence
# --------------------------------------------------

def get_prediction_confidence(image):

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(tensor)

        probabilities = torch.softmax(output, dim=1)

        prediction = output.argmax(dim=1).item()

        confidence = probabilities[0, prediction].item()

    return prediction, confidence


# --------------------------------------------------
# 6. Function for regional blur
# --------------------------------------------------

def create_region_versions(image):

    width, height = image.size

    top_blurred = image.copy()

    bottom_blurred = image.copy()

    # Blur top half

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

    # Blur bottom half

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

def run_group(indices):

    top_changes = []
    bottom_changes = []

    for index in indices:

        image, true_label = test_dataset[index]

        # IMPORTANT:
        # test_dataset returns a transformed tensor,
        # so convert it back approximately to an image.

        image = image.clone()

        mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(3, 1, 1)

        image = image * std + mean

        image = torch.clamp(image, 0, 1)

        image = transforms.ToPILImage()(image)

        # Original prediction

        _, original_confidence = get_prediction_confidence(
            image
        )

        # Regional perturbations

        top_blurred, bottom_blurred = create_region_versions(
            image
        )

        _, top_confidence = get_prediction_confidence(
            top_blurred
        )

        _, bottom_confidence = get_prediction_confidence(
            bottom_blurred
        )

        # Confidence change

        top_change = original_confidence - top_confidence

        bottom_change = (
            original_confidence - bottom_confidence
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

correct_top, correct_bottom = run_group(
    correct_indices
)

wrong_top, wrong_bottom = run_group(
    wrong_indices
)


# --------------------------------------------------
# 9. Calculate averages
# --------------------------------------------------

def average(values):

    return sum(values) / len(values)


print("=" * 50)
print("RESULTS")
print("=" * 50)

print()

print("CORRECTLY CLASSIFIED MOUNTAINS")

print(
    f"Top blur confidence drop: "
    f"{average(correct_top):.2%}"
)

print(
    f"Bottom blur confidence drop: "
    f"{average(correct_bottom):.2%}"
)

print()

print("MISCLASSIFIED MOUNTAINS")

print(
    f"Top blur confidence drop: "
    f"{average(wrong_top):.2%}"
)

print(
    f"Bottom blur confidence drop: "
    f"{average(wrong_bottom):.2%}"
)