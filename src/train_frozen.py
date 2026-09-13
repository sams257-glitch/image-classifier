import torch
import torch.nn as nn
import torch.optim as optim

from dataset import train_loader
from model_frozen import create_model

EPOCHS = 3
LEARNING_RATE = 0.001

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = create_model().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
)

print("Device:", device)

for epoch in range(EPOCHS):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- Loss: {running_loss / len(train_loader):.4f} "
        f"- Train Accuracy: {accuracy:.2f}%"
    )

torch.save(
    model.state_dict(),
    "models/resnet18_frozen.pth"
)

print("Model saved to models/resnet18_frozen.pth")