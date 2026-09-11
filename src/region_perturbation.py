import torch
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

from dataset import test_dataset
from model import create_model
from torchvision import transforms


# --------------------------------------------------
# 1. Load the trained model
# --------------------------------------------------

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


# --------------------------------------------------
# 2. Load our selected image
# --------------------------------------------------

INDEX = 1726

image_path, true_label = test_dataset.samples[INDEX]

original = Image.open(image_path).convert("RGB")


# --------------------------------------------------
# 3. Create regional perturbations
# --------------------------------------------------

width, height = original.size

top_blurred = original.copy()
bottom_blurred = original.copy()

# Blur the top half
top_half = original.crop(
    (0, 0, width, height // 2)
)

top_half = top_half.filter(
    ImageFilter.GaussianBlur(radius=8)
)

top_blurred.paste(
    top_half,
    (0, 0)
)

# Blur the bottom half
bottom_half = original.crop(
    (0, height // 2, width, height)
)

bottom_half = bottom_half.filter(
    ImageFilter.GaussianBlur(radius=8)
)

bottom_blurred.paste(
    bottom_half,
    (0, height // 2)
)


images = {
    "Original": original,
    "Top Blurred": top_blurred,
    "Bottom Blurred": bottom_blurred,
}


# --------------------------------------------------
# 4. Prepare images for the model
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
# 5. Get predictions
# --------------------------------------------------

results = []

for name, image in images.items():

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)

    probabilities = torch.softmax(output, dim=1)

    predicted_class = output.argmax(dim=1).item()

    confidence = probabilities[0, predicted_class].item()

    results.append({
        "name": name,
        "image": image,
        "prediction": predicted_class,
        "confidence": confidence,
    })


# --------------------------------------------------
# 6. Print results
# --------------------------------------------------

print(
    f"True class: "
    f"{test_dataset.classes[true_label]}"
)

print()

for result in results:

    print(
        f"{result['name']:14s} → "
        f"{test_dataset.classes[result['prediction']]:10s} "
        f"({result['confidence']:.2%})"
    )


# --------------------------------------------------
# 7. Display results
# --------------------------------------------------

fig, axes = plt.subplots(
    1,
    len(results),
    figsize=(15, 5)
)

for ax, result in zip(axes, results):

    ax.imshow(result["image"])

    ax.set_title(
        f"{result['name']}\n"
        f"Pred: "
        f"{test_dataset.classes[result['prediction']]}\n"
        f"Confidence: "
        f"{result['confidence']:.1%}"
    )

    ax.axis("off")


plt.tight_layout()

plt.savefig(
    "results/region_perturbation.png"
)

plt.show()