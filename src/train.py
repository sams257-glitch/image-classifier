import torch
import torch.nn as nn
import torch.optim as optim

from dataset import train_loader
from model import create_model


# -------------------------
# Configuration
# -------------------------

EPOCHS = 3
LEARNING_RATE = 0.001

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using device:", device)


# -------------------------
# Model
# -------------------------

model = create_model().to(device)


# -------------------------
# Loss and optimizer
# -------------------------

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# -------------------------
# Training
# -------------------------

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Statistics
        running_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- Loss: {epoch_loss:.4f} "
        f"- Accuracy: {epoch_accuracy:.2f}%"
    )


# Save model

torch.save(
    model.state_dict(),
    "models/resnet18_baseline.pth"
)

print("Model saved to models/resnet18_baseline.pth")