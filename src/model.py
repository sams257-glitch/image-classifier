import torch
from torchvision import models


NUM_CLASSES = 6


def create_model(pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None

    model = models.resnet18(weights=weights)

    # Replace ImageNet's 1000-class classifier
    # with our 6-class classifier.
    model.fc = torch.nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    return model


if __name__ == "__main__":
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model = create_model().to(device)

    print("Device:", device)
    print(model.fc)