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
# 2. Load our selected test image
# --------------------------------------------------

INDEX = 1726

image_path, true_label = test_dataset.samples[INDEX]

original = Image.open(image_path).convert("RGB")


# --------------------------------------------------
# 3. Create controlled spatial changes
# --------------------------------------------------

images = {
    "Original": original,
    "Light Blur": original.filter(ImageFilter.GaussianBlur(radius=2)),
    "Strong Blur": original.filter(ImageFilter.GaussianBlur(radius=6)),
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
        f"{result['name']:12s} → "
        f"{test_dataset.classes[result['prediction']]:10s} "
        f"({result['confidence']:.2%})"
    )


# --------------------------------------------------
# 7. Display images
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
    "results/spatial_perturbation.png"
)

plt.show()