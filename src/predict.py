import torch
from PIL import Image
from torchvision import transforms

from src.model import create_model


CLASS_NAMES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street",
]

MODEL_PATH = "models/resnet18_baseline.pth"

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# Same preprocessing used for test images.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


model = create_model().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
    )
)

model.eval()


def predict(image_path, top_k=3):
    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    top_probabilities, top_indices = torch.topk(
        probabilities,
        top_k,
        dim=1,
    )

    results = []

    for probability, index in zip(
        top_probabilities[0],
        top_indices[0],
    ):
        results.append({
            "class": CLASS_NAMES[index.item()],
            "confidence": probability.item(),
        })

    return results


if __name__ == "__main__":
    image_path = input("Enter image path: ")

    results = predict(image_path)

    for result in results:
        print(
            f"{result['class']}: "
            f"{result['confidence'] * 100:.2f}%"
        )