# Natural Scene Image Classifier

A computer vision project for classifying natural scene images into six categories using transfer learning with ResNet-18.

## Classes

- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

## Model

The project uses a pretrained ResNet-18 model with the final classification layer replaced for six classes.

Training was performed using PyTorch with Apple Silicon MPS acceleration.

## Baseline

The initial ResNet-18 model achieved:

- Training accuracy: 89.82%
- Test accuracy: 89.47%

The main classification difficulties were:

- Mountain vs Glacier
- Buildings vs Street
- Mountain vs Sea

The mountain class had particularly low recall at 75%.

## Error Analysis

Instead of only measuring accuracy, I analyzed model failures using:

- Confusion matrices
- High-confidence incorrect predictions
- Image perturbation experiments
- Spatial region perturbations

For a mountain image incorrectly classified as sea, global brightness and saturation changes did not change the predicted class, while spatial blurring reduced confidence.

Regional perturbation experiments were then performed across correctly and incorrectly classified mountain images.

The results showed a statistically significant difference in regional sensitivity between correctly classified and misclassified mountain images.

For the upper region:

- Mann–Whitney U p < 0.001
- Rank-biserial correlation = 0.692

For the lower region:

- Mann–Whitney U p < 0.001
- Rank-biserial correlation = 0.340

These results suggest that incorrectly classified mountain images respond differently to the removal of regional visual information. This does not by itself prove that the model relies on a specific region, since differences may also arise from image composition, ambiguous visual cues, or label noise.

## Data Augmentation

I introduced training-time augmentation using:

- Random horizontal flipping
- Random rotation
- Random brightness/contrast/saturation changes

The test set remained unchanged.

After augmentation:

- Test accuracy: 89.03%
- Mountain recall: 79%

Overall accuracy decreased slightly compared with the baseline, but the distribution of mountain errors changed substantially.

Mountain → Sea errors decreased from 37 to 8, while Mountain → Glacier errors increased from 90 to 101.

Among mountain test images, the augmented model changed 97 predictions:

- 53 became correct
- 33 became incorrect

This suggests that augmentation changed the model's decision boundary rather than simply improving performance across every class.

## Project Goal

The goal of this project is not simply to maximize classification accuracy, but to investigate how a vision model behaves when faced with visually similar scenes and how training choices affect its failure patterns.

## Tech Stack

- Python
- PyTorch
- Torchvision
- ResNet-18
- Scikit-learn
- Matplotlib
- Pillow
- Apple Silicon MPS