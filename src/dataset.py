from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


DATA_DIR = Path("data")

train_dir = DATA_DIR / "seg_train" / "seg_train"
test_dir = DATA_DIR / "seg_test" / "seg_test"


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


train_dataset = datasets.ImageFolder(
    train_dir,
    transform=transform,
)

test_dataset = datasets.ImageFolder(
    test_dir,
    transform=transform,
)


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
)


if __name__ == "__main__":
    print("Classes:", train_dataset.classes)
    print("Number of training images:", len(train_dataset))
    print("Number of test images:", len(test_dataset))

    images, labels = next(iter(train_loader))

    print("Batch image shape:", images.shape)
    print("Batch label shape:", labels.shape)
