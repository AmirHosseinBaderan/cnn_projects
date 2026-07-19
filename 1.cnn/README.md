# Project 1: CIFAR-10 Image Classification

## What is this project?

This project teaches you how to build an **image classification system** using Convolutional Neural Networks (CNNs). We use the famous **CIFAR-10 dataset**, which contains 60,000 small images (32x32 pixels) across 10 different categories:

- Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

The goal is to train a neural network that can look at an image and correctly identify which of these 10 categories it belongs to.

## What will you learn?

By completing this project, you will understand:

1. **How CNNs work** - The architecture that powers modern computer vision
2. **Data loading and augmentation** - How to prepare images for training
3. **Training loops** - The step-by-step process of teaching a neural network
4. **Model evaluation** - How to measure if your model is learning correctly
5. **Transfer learning** - Using pre-trained models like ResNet and EfficientNet

## Project Structure

```
1.cnn/
├── config.py              # All settings in one place
├── train.py               # Main training script
├── predict.py             # Load model and make predictions
├── datasets/
│   └── image_dataset.py   # Load and prepare CIFAR-10 data
├── models/
│   ├── cnn.py             # Our custom CNN architecture
│   ├── resnet.py          # ResNet18 wrapper
│   ├── efficientnet.py    # EfficientNet-B0 wrapper
│   └── factory.py         # Choose which model to use
├── engine/
│   ├── trainer.py         # Training logic for one epoch
│   └── evaluator.py       # Validation logic
└── utils/
    ├── checkpoint.py      # Save and load model weights
    ├── logger.py          # Print training progress
    ├── metrics.py         # Calculate accuracy
    └── visualize.py       # Draw training graphs
```

## How It Works - The Big Picture

### 1. Data Preparation

Before training, we need to prepare our images. Think of this like preparing ingredients before cooking:

```python
# From datasets/image_dataset.py
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),      # Slightly crop the image
    transforms.RandomHorizontalFlip(),         # Flip image horizontally
    transforms.RandomRotation(10),             # Rotate slightly
    transforms.ColorJitter(...),               # Change colors a bit
    transforms.ToTensor(),                     # Convert to numbers
    transforms.Normalize(...),                 # Standardize the numbers
])
```

**Why do we do this?**
- **RandomCrop**: Teaches the model to recognize objects even if they're not perfectly centered
- **RandomHorizontalFlip**: Doubles our training data by using mirror images
- **Normalization**: Makes training more stable by keeping pixel values in a consistent range

### 2. The CNN Model

Our custom CNN (`models/cnn.py`) has this structure:

```
Input Image (3, 32, 32)
    ↓
[Conv2d: 3→32 channels] → ReLU → MaxPool
    ↓
[Conv2d: 32→64 channels] → ReLU → MaxPool
    ↓
Flatten → [Linear: 4096→128] → ReLU
    ↓
[Linear: 128→10] → Output (10 class scores)
```

**Let me explain each part:**

- **Conv2d (Convolution)**: Like a sliding window that looks for patterns (edges, textures, shapes)
- **ReLU**: Turns negative numbers to zero, adding non-linearity
- **MaxPool**: Takes the maximum value in a region, reducing size while keeping important features
- **Linear (Fully Connected)**: Makes the final decision based on all extracted features

### 3. The Training Process

Training happens in `train.py`. Here's what happens each epoch:

```python
for epoch in range(NUM_EPOCHS):
    # 1. Training phase
    model.train()  # Tell PyTorch we're training
    for images, labels in train_loader:
        outputs = model(images)           # Forward pass
        loss = criterion(outputs, labels) # Calculate error
        optimizer.zero_grad()             # Clear old gradients
        loss.backward()                   # Calculate new gradients
        optimizer.step()                  # Update weights

    # 2. Validation phase
    model.eval()  # Tell PyTorch we're evaluating
    with torch.no_grad():
        for images, labels in valid_loader:
            # Just check accuracy, don't update weights
```

**The key insight**: Training is like a feedback loop. The model makes predictions, we calculate how wrong it is (loss), and then we adjust the model's weights to be less wrong next time.

### 4. Model Selection with Factory Pattern

We support three different models through the `ModelFactory`:

```python
# From models/factory.py
model = ModelFactory.create(
    model_name="cnn",        # or "resnet18", "efficientnet_b0"
    num_classes=10,
    pretrained=True          # Use pre-trained weights?
)
```

- **cnn**: Our custom-built model (learn from scratch)
- **resnet18**: A famous architecture with skip connections (transfer learning)
- **efficientnet_b0**: A modern, efficient architecture (transfer learning)

## How to Run

### Step 1: Install Dependencies

```bash
pip install torch torchvision matplotlib scikit-learn
```

### Step 2: Train the Model

```bash
cd 1.cnn
python train.py
```

This will:
1. Download CIFAR-10 dataset (if not already present)
2. Train for 20 epochs
3. Save the best model to `checkpoints/best_model.pth`
4. Show training/validation graphs

### Step 3: Make Predictions

```bash
python predict.py
```

## Key Concepts Explained

### What is a Convolution?

Imagine you have a small flashlight (the kernel) that you slide over an image. At each position, you multiply the flashlight area with the image and sum the results. This helps detect features like edges.

### What is Backpropagation?

When the model makes a mistake, backpropagation calculates how much each weight contributed to the error, then adjusts the weights to reduce future errors. It's like learning from your mistakes!

### What is Cross-Entropy Loss?

This measures how far the model's predictions are from the true labels. Lower loss = better predictions.

### What is Adam Optimizer?

Adam is the algorithm that updates the model's weights. It's smarter than basic gradient descent because it adapts the learning rate for each parameter.

## Expected Results

With our custom CNN, you should achieve around **70-75% accuracy** on CIFAR-10 after 20 epochs. With ResNet18 or EfficientNet (using transfer learning), you can reach **85-90% accuracy**.

## Troubleshooting

- **Out of Memory?** Reduce `BATCH_SIZE` in `config.py`
- **Training too slow?** Use a GPU or reduce `NUM_EPOCHS`
- **Low accuracy?** Try more epochs, data augmentation, or a different model

## Next Steps

After completing this project, you'll be ready for:
- Project 2: Working with custom datasets and more advanced training techniques
- Project 3: Face recognition using metric learning
