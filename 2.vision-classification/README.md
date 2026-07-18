# Project 2: Scene Classification (Intel Image Dataset)

## What is this project?

This project teaches you how to build an **image classification system for natural scenes**. We use the **Intel Image Classification dataset**, which contains thousands of images across 6 scene categories:

- **Buildings** - Urban architecture and structures
- **Forest** - Dense woodland and trees
- **Glacier** - Ice formations and snowy mountains
- **Mountain** - Rocky peaks and high terrain
- **Sea** - Ocean and coastal waters
- **Street** - Roads and urban streets

The goal is to train a neural network that can look at any natural scene image and correctly identify its category.

## What will you learn?

By completing this project, you will understand:

1. **Custom Dataset Creation** - How to load images from your own folder structure
2. **Advanced Training Techniques** - Early stopping, learning rate scheduling
3. **Model Architecture Design** - Building CNNs from scratch for specific tasks
4. **Evaluation Metrics** - Confusion matrices and classification reports
5. **Checkpoint Management** - Saving and resuming training

## Project Structure

```
2.vision-classification/
├── train.py                # Main training script
├── predict.py              # Load model and predict
├── evaluator.py            # Validation logic
├── test.py                 # Quick model test
├── datasets/
│   └── intel_dataset.py    # Custom dataset class
├── models/
│   └── simple_cnn.py       # CNN architecture
├── transforms/
│   └── transforms.py       # Image preprocessing
├── utils/
│   ├── checkpoint.py       # Save/load model states
│   ├── logger.py           # Training logs
│   └── metrics.py          # Confusion matrix, reports
└── data/
    ├── raw/
    │   ├── seg_train/      # Training images (organized by class)
    │   ├── seg_test/       # Test images
    │   └── seg_pred/       # Images for prediction
    └── checkpoints/        # Saved models
```

## How It Works - The Big Picture

### 1. Understanding the Data

The Intel dataset is organized in folders by class:

```
data/raw/seg_train/
├── buildings/    (thousands of .jpg files)
├── forest/
├── glacier/
├── mountain/
├── sea/
└── street/
```

Our custom `IntelDataset` class reads this structure automatically:

```python
# From datasets/intel_dataset.py
class IntelDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        
        # Build list of (image_path, label) pairs
        self.images = []
        for cls_name in self.classes:
            for img_name in os.listdir(cls_path):
                self.images.append((img_path, self.class_to_idx[cls_name]))
```

**Why is this important?** In real projects, you rarely get neat pre-packaged datasets. Learning to create custom dataset classes is an essential skill.

### 2. The Model Architecture

Our `SimpleCNN` is designed for 150x150 images:

```
Input Image (3, 150, 150)
    ↓
[Conv2d: 3→32, kernel=3] → ReLU → MaxPool(2x2)
    ↓  (output: 32, 75, 75)
[Conv2d: 32→64, kernel=3] → ReLU → MaxPool(2x2)
    ↓  (output: 64, 37, 37)
Flatten → (64 × 37 × 37 = 87,616 values)
    ↓
[Linear: 87616→512] → ReLU
    ↓
[Linear: 512→6] → Output (6 class scores)
```

**Key design decisions:**
- **Two convolutional layers**: Enough to learn scene features without overfitting
- **Kernel size 3**: Small kernels capture fine details
- **512 hidden units**: A good balance for this dataset size
- **6 output classes**: One for each scene type

### 3. Advanced Training Features

This project introduces techniques that make training more robust:

#### Early Stopping

```python
# From train.py
if valid_loss < best_loss:
    best_loss = valid_loss
    early_stopping_counter = 0
else:
    early_stopping_counter += 1

if early_stopping_counter >= PATIENCE:
    print("Early stopping - model stopped improving")
    break
```

**What is it?** If the validation loss doesn't improve for 5 epochs, we stop training. This prevents overfitting (when the model memorizes training data but fails on new data).

#### Learning Rate Scheduling

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)
```

**What is it?** When the model stops improving, we reduce the learning rate by half. This helps the model make finer adjustments to find the optimal weights.

### 4. The Training Loop

```python
for epoch in range(start_epoch, EPOCHS):
    # Training phase
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()      # Clear gradients
        outputs = model(images)    # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()            # Backpropagation
        optimizer.step()           # Update weights
    
    # Validation phase
    valid_loss, valid_accuracy, _, _ = validate(...)
    
    # Save best model
    if valid_accuracy > best_accuracy:
        save_checkpoint(...)
    
    # Check early stopping
    if early_stopping_counter >= PATIENCE:
        break
```

### 5. Evaluation and Metrics

After training, we evaluate using:

```python
# Confusion Matrix - shows which classes are confused
cm = confusion_matrix(all_labels, all_predictions)

# Classification Report - precision, recall, F1-score per class
report = classification_report(all_labels, all_predictions)
```

**Why these matter:**
- **Confusion Matrix**: Shows if the model confuses "glacier" with "mountain"
- **Precision**: Of all predicted "forest" images, how many were actually forest?
- **Recall**: Of all actual "forest" images, how many did we find?

## How to Run

### Step 1: Prepare Data

The dataset should be organized as:
```
data/raw/
├── seg_train/
│   ├── buildings/
│   ├── forest/
│   ├── glacier/
│   ├── mountain/
│   ├── sea/
│   └── street/
└── seg_test/
    └── (same structure)
```

### Step 2: Train the Model

```bash
cd 2.vision-classification
python train.py
```

This will:
1. Load images from `data/raw/seg_train`
2. Train for up to 10 epochs (with early stopping)
3. Save the best model to `checkpoints/model.pth`
4. Log progress to console

### Step 3: Make Predictions

```bash
python predict.py
```

This loads a test image and shows the top 3 predictions with confidence scores.

## Key Concepts Explained

### What is a Dataset Class?

A `Dataset` class is a Python object that knows how to:
1. Find your data files
2. Load them into memory
3. Apply transformations
4. Return one item at a time when asked

PyTorch's `DataLoader` uses this to efficiently batch and shuffle data.

### What is Early Stopping?

Imagine studying for an exam. You practice with past papers (training) and take mock tests (validation). If your mock test scores stop improving, continuing to study might actually make you worse (overfitting). Early stopping is like knowing when to stop studying!

### What is a Confusion Matrix?

A table showing:
- Rows: True labels
- Columns: Predicted labels
- Diagonal: Correct predictions
- Off-diagonal: Mistakes

If row "glacier" and column "mountain" has a high number, the model often confuses them.

## Expected Results

With this architecture, you should achieve **80-85% accuracy** on the test set. The model typically performs best on:
- **Street** and **Buildings** (distinctive features)
- **Sea** (unique blue color)

And may struggle with:
- **Glacier vs Mountain** (both have white/gray rocky features)
- **Forest vs Mountain** (both have green/brown natural features)

## Troubleshooting

- **Import errors?** Make sure you're running from the `2.vision-classification` directory
- **CUDA out of memory?** Reduce `BATCH_SIZE` in `train.py`
- **Low accuracy?** Try more data augmentation or train for more epochs
- **Overfitting?** Increase dropout or use more data augmentation

## Next Steps

After this project, you'll be ready for:
- Project 3: Face recognition using metric learning and triplet loss
