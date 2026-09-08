import torch

from dataset import train_loader
from model import create_model


device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

model = create_model().to(device)
model.eval()

images, labels = next(iter(train_loader))

images = images.to(device)

with torch.no_grad():
    outputs = model(images)

print("Device:", device)
print("Input shape:", images.shape)
print("Output shape:", outputs.shape)
print("Predicted classes:", outputs.argmax(dim=1))
