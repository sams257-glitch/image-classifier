import torch
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

from dataset import test_dataset
from model import create_model


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
# 3. Create controlled visual changes
# --------------------------------------------------

images = {
    "Original": original,

    "Brighter": ImageEnhance.Brightness(
        original
    ).enhance(1.5),

    "Darker": ImageEnhance.Brightness(
        original
    ).enhance(0.5),

    "More Saturated": ImageEnhance.Color(
        original
    ).enhance(1.8),

    "Less Saturated": ImageEnhance.Color(
        original
    ).enhance(0.3),
}


# --------------------------------------------------
# 4. Prepare images for the model
# --------------------------------------------------

from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# --------------------------------------------------
# 5. Ask the model for a prediction
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
# 6. Print the results
# --------------------------------------------------

print(f"True class: {test_dataset.classes[true_label]}")
print()

for result in results:

    print(
        f"{result['name']:16s} → "
        f"{test_dataset.classes[result['prediction']]:10s} "
        f"({result['confidence']:.2%})"
    )


# --------------------------------------------------
# 7. Display all versions
# --------------------------------------------------

fig, axes = plt.subplots(
    1,
    len(results),
    figsize=(20, 5)
)

for ax, result in zip(axes, results):

    ax.imshow(result["image"])

    ax.set_title(
        f"{result['name']}\n"
        f"Pred: {test_dataset.classes[result['prediction']]}\n"
        f"Confidence: {result['confidence']:.1%}"
    )

    ax.axis("off")


plt.tight_layout()

plt.savefig(
    "results/perturbation_test.png"
)

plt.show()