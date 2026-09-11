from PIL import Image
import matplotlib.pyplot as plt

from dataset import test_dataset


INDEX = 1726

image_path, label = test_dataset.samples[INDEX]

image = Image.open(image_path).convert("RGB")

print("Image path:", image_path)
print("True class:", test_dataset.classes[label])

plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.title(
    f"True class: {test_dataset.classes[label]}\n"
    f"Test index: {INDEX}"
)
plt.axis("off")

plt.savefig("results/selected_test_image.png")
plt.show()