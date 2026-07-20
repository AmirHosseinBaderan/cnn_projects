# OCR Project — Scene Text Recognition with CRNN

This project implements **Optical Character Recognition (OCR)** for recognizing text in natural scene images. It uses a **CRNN (Convolutional Recurrent Neural Network)** architecture trained with **CTC (Connectionist Temporal Classification)** loss.

This is the fourth and most advanced project in the series. It combines CNNs, RNNs, and sequence-to-sequence learning into a single end-to-end trainable system.

---

## What is OCR?

**Optical Character Recognition (OCR)** is the process of converting images containing text into machine-readable text. In this project, we focus on **scene text recognition** — reading text from natural images (like street signs, product labels, etc.).

Unlike simple digit recognition (like MNIST), scene text recognition is hard because:
- Text can be in any font, size, or orientation
- Backgrounds are complex and noisy
- Words have variable lengths
- There is no fixed "character bounding box" — we don't know where each character starts and ends

This project solves these problems using a **CRNN + CTC** approach.

---

## How It Works — The Big Picture

Traditional OCR would require you to first find each character (segmentation) and then classify it. This project does **not** do segmentation. Instead, it uses an end-to-end approach:

```
Image of word "HELLO"
        │
        ▼
  ┌─────────────┐
  │    CNN       │  Extract visual features
  │ (ResNet-like)│  Image: 32×128 → Features: (B, 512, 1, 32)
  └──────┬───────┘
         │
         ▼
  ┌─────────────┐
  │  Converter   │  Reshape 2D features → 1D sequence
  │              │  (B, 512, 1, 32) → (B, 32, 512)
  └──────┬───────┘
         │
         ▼
  ┌─────────────┐
  │ BiLSTM       │  Model sequence dependencies
  │              │  (B, 32, 512) → (B, 32, 512)
  └──────┬───────┘
         │
         ▼
  ┌─────────────┐
  │ Classifier   │  Predict character at each position
  │  (Linear)    │  (B, 32, 512) → (B, 32, 63) = logits
  └──────┬───────┘
         │
         ▼
  ┌─────────────┐
  │  CTC Loss    │  Align predictions with text without knowing alignment
  └─────────────┘
```

The magic is in **CTC loss**. It lets the network output a sequence of 32 predictions (one for each "column" of the image), but the target text "HELLO" has only 5 characters. CTC learns to "collapse" the 32 outputs into the 5 characters by inserting a special **blank** token between repeated characters.

---

## How This Project Was Created

### 1. Data Preparation
- Source: **Synth90k** dataset style — pairs of word images and their text labels
- Images: Grayscale word images resized to **32×128** pixels
- Labels: Text strings (e.g., "HELLO")
- Split: 90% training, 10% validation (via `dataset/split.py`)
- Vocabulary: Built from all labels using `dataset/build_vocab.py`, saved to `resources/vocab.json`

### 2. Architecture Design
- **CNN**: 7 convolutional layers progressively reducing spatial dimensions while increasing channels. Final output has height=1, so width=32 becomes our "sequence length."
- **LSTM**: 2-layer bidirectional LSTM to capture left-to-right and right-to-left context in the text.
- **Classifier**: Simple linear layer mapping LSTM outputs to character probabilities.
- **Loss**: CTC loss handles variable-length outputs without needing character-level annotations.

### 3. Training Pipeline
Built incrementally:
1. Start with `main.py` to verify each module works independently (forward pass, backward pass)
2. Assemble full `CRNN` model and test end-to-end
3. Build `train.py` orchestrating dataset, model, optimizer, and trainer
4. Add TensorBoard logging, checkpointing, and validation metrics

---

## How to Use This Project

### Prerequisites
```bash
pip install torch torchvision opencv-python pillow matplotlib tensorboard tqdm sympy
```

### Step 1: Prepare Data
Place your images in `data/images/` and create annotation files:
- `data/train.txt` — each line: `<image_name> <text_label>`
- `data/valid.txt` — same format for validation

Build vocabulary and split data:
```bash
python -c "from dataset.build_vocab import build_vocabulary; build_vocabulary('data/labels.txt', 'resources/vocab.json')"
python -c "from dataset.split import create_train_valid_split; create_train_valid_split('data/labels.txt', 'data')"
```

### Step 2: Train
```bash
python train.py
```
This will:
- Load data and vocabulary
- Build the CRNN model
- Train for 50 epochs (configurable in `config.py`)
- Log metrics to TensorBoard in `runs/`
- Save best model to `checkpoints/best.pt` and last model to `checkpoints/last.pt`

### Step 3: Monitor Training
```bash
tensorboard --logdir runs
```

### Step 4: Test / Predict
`predict.py` is a placeholder for future inference. The current way to test is through the validation loop in `trainer/validate.py`, which decodes predictions after each epoch.

### Step 5: Verify Architecture
```bash
python main.py
```
This runs a sanity check: it builds the model, does a forward pass on real data, computes loss, and verifies gradients flow with `loss.backward()`.

---

## Project Structure

```
4.ocr/
├── config.py                     # Hyperparameters and paths
├── main.py                       # Architecture sanity check (end-to-end test)
├── train.py                      # Main training entry point
├── predict.py                    # Placeholder for future inference script
│
├── data/
│   ├── train.txt                 # Training annotations: image_name + text
│   ├── valid.txt                 # Validation annotations
│   ├── labels.txt                # All labels (used to build vocabulary)
│   └── images/                   # Grayscale word images (32×128)
│
├── resources/
│   └── vocab.json                # Character-to-index mapping
│
├── runs/                         # TensorBoard event logs
├── checkpoints/                  # Saved model weights (best.pt, last.pt)
│
├── models/
│   ├── recognizer.py             # CRNN: full model (CNN + RNN + Classifier)
│   ├── detector.py               # Empty placeholder (future: text detection)
│   └── modules/
│       ├── cnn.py                # CNNFeatureExtractor: 7-layer conv backbone
│       ├── sequence.py           # SequenceConverter + BidirectionalLSTM
│       └── classifier.py         # CTCClassifier: final linear projection
│
├── dataset/
│   ├── synth90k.py               # Synth90kDataset: loads image+text pairs
│   ├── vocabulary.py             # Vocabulary: character encoding/decoding
│   ├── collate.py                # CTCCollate: custom batch collation for CTC
│   ├── build_vocab.py            # Utility to generate vocab.json from labels
│   └── split.py                  # Utility to split labels into train/valid
│
├── preprocessing/
│   ├── transforms.py             # Image resize, tensor conversion, normalization
│   └── image.py                  # Empty placeholder (future preprocessing)
│
├── decoder/
│   └── greedy.py                 # GreedyDecoder: argmax + blank/duplicate removal
│
├── trainer/
│   ├── trainer.py                # Trainer: orchestrates epochs, logging, checkpoints
│   ├── train_one_epoch.py        # Single epoch training loop with progress bar
│   ├── validate.py               # Validation loop + prediction decoding
│   ├── loss.py                   # CTCLossWrapper: wraps nn.CTCLoss
│   ├── metrics.py                # OCRMetrics: compares predictions vs ground truth
│   ├── logger.py                 # TensorBoardLogger: logs scalars, images, text
│   └── checkpoint.py             # CheckpointManager: save/load model states
│
└── utils/
    ├── logger.py                 # Python logging configuration
    └── visualize.py              # show_batch: matplotlib grid for debugging
```

---

## Detailed File & Class Reference

### Configuration

#### `config.py`
**Class `Config`**
- Central configuration container with class attributes.
- `DEVICE`: "cuda" if available, else "cpu"
- `IMAGE_HEIGHT=32`, `IMAGE_WIDTH=128`: Standard input size for word images
- `BATCH_SIZE=32`, `LEARNING_RATE=1e-3`, `EPOCHS=50`
- `NUM_WORKERS=4`, `PIN_MEMORY`: DataLoader optimizations
- `CHECKPOINT_DIR="checkpoints"`, `LOG_DIR="runs"`

---

### Models

#### `models/recognizer.py`
**Class `CRNN(nn.Module)`**
The complete model. It composes three sub-modules:
- `self.cnn` → `CNNFeatureExtractor`: extracts features from images
- `self.converter` → `SequenceConverter`: reshapes features for RNN
- `self.sequence` → `BidirectionalLSTM`: models sequence dependencies
- `self.classifier` → `CTCClassifier`: outputs character logits

**`forward(self, x)`**
1. Pass image through CNN → feature map `(B, 512, 1, 32)`
2. Convert to sequence `(B, 32, 512)`
3. Pass through BiLSTM → `(B, 32, 512)`
4. Classify each timestep → `(B, 32, num_classes)` logits
5. Return dict: `{"logits": logits, "input_lengths": [32]*batch_size}`

---

#### `models/modules/cnn.py`
**Class `CNNFeatureExtractor(nn.Module)`**
7-layer convolutional backbone. Each layer progressively reduces height while increasing channel depth.

Layer-by-layer:
| Layer | Output Size | Operation |
|-------|-------------|-----------|
| Input | 32×128 | Grayscale image |
| Conv1 | 16×64 | Conv2d(1→64, k=3) + ReLU + MaxPool(2,2) |
| Conv2 | 8×32 | Conv2d(64→128, k=3) + ReLU + MaxPool(2,2) |
| Conv3 | 8×32 | Conv2d(128→256, k=3) + BN + ReLU |
| Conv4 | 4×32 | Conv2d(256→256, k=3) + ReLU + MaxPool(2,1) |
| Conv5 | 4×32 | Conv2d(256→512, k=3) + BN + ReLU |
| Conv6 | 2×32 | Conv2d(512→512, k=3) + ReLU + MaxPool(2,1) |
| Conv7 | 1×32 | Conv2d(512→512, k=(2,1)) + ReLU |

Final output: `(Batch, 512, 1, 32)` — 512 feature channels, 1 pixel height, 32 width positions.

---

#### `models/modules/sequence.py`
**Class `SequenceConverter(nn.Module)`**
Reshapes the 2D CNN output into a 1D sequence for the LSTM.
- Input: `(B, C, H, W)` = `(B, 512, 1, 32)`
- Squeeze height: `(B, 512, 32)`
- Permute: `(B, 32, 512)` — now 32 timesteps, each with 512 features

**Class `BidirectionalLSTM(nn.Module)`**
2-layer bidirectional LSTM.
- `input_size=512`, `hidden_size=256`, `num_layers=2`
- `bidirectional=True` → output is 512 (256 forward + 256 backward)
- `dropout=0.2` between layers
- `batch_first=True` → input shape `(B, T, C)`
- Output: `(B, 32, 512)`

---

#### `models/modules/classifier.py`
**Class `CTCClassifier(nn.Module)`**
Simple linear projection from LSTM hidden states to character classes.
- `nn.Linear(512, num_classes)`
- Input: `(B, 32, 512)`
- Output: `(B, 32, num_classes)` raw logits

---

### Dataset

#### `dataset/synth90k.py`
**Class `Synth90kDataset(Dataset)`**
PyTorch Dataset for loading image-text pairs.
- `__init__`: Parses annotation file (format: `image_name text_label`) into `self.samples`
- `__len__`: Returns number of samples
- `__getitem__`: Opens image as grayscale PIL, applies transforms, encodes text label to integer indices via Vocabulary
- Returns: `{"image": Tensor, "label": str, "target": List[int]}`

---

#### `dataset/vocabulary.py`
**Class `Vocabulary`**
Manages the character-to-index mapping.
- `__init__`: Loads `vocab.json`. Builds `char_to_idx` (index starts at 1, since 0 is reserved for CTC blank) and `idx_to_char`.
- `num_classes` (property): Returns `len(characters) + 1` (includes blank token)
- `encode(text)`: Converts string → list of integer indices
- `decode(indices)`: Converts list of indices → string, removing blank tokens and consecutive duplicates

**Example:**
```python
vocab = Vocabulary("resources/vocab.json")
encoded = vocab.encode("HELLO")   # [17, 14, 21, 21, 29]
decoded = vocab.decode(encoded)   # "HELLO"
```

---

#### `dataset/collate.py`
**Class `CTCCollate`**
Custom collate function for the DataLoader. CTC requires specific tensor formatting:
- Stacks images into a single tensor `(B, 1, 32, 128)`
- Concatenates all target sequences into one flat tensor
- Records `target_lengths` (how many characters per sample)
- Returns: `{"images": Tensor, "targets": Tensor, "labels": List[str], "target_lengths": Tensor}`

This is crucial because CTC loss needs flattened targets and their lengths.

---

#### `dataset/build_vocab.py`
**Function `build_vocabulary(labels_file, output_file)`**
Scans all text labels in the annotation file, collects unique characters, sorts them, and writes `vocab.json` with:
```json
{
    "blank": 0,
    "characters": ["0", "1", ..., "A", "B", ..., "a", "b", ...]
}
```

---

#### `dataset/split.py`
**Function `create_train_valid_split(labels_file, output_dir, train_rate=0.9, seed=42)`**
Shuffles all samples and splits them into `train.txt` (90%) and `valid.txt` (10%).

---

### Preprocessing

#### `preprocessing/transforms.py`
Defines image preprocessing pipelines:
- `train_transform`: Resize(32×128) → ToTensor → Normalize(mean=0.5, std=0.5)
- `valid_transform`: Same as train (no augmentation applied)

Images are normalized to the range [-1, 1].

---

### Decoder

#### `decoder/greedy.py`
**Class `GreedyDecoder`**
Converts raw logits into text strings during inference.
- `decode(logits)`:
  1. `argmax` over class dimension → most likely character index at each timestep
  2. Remove all blank tokens (index 0)
  3. Remove consecutive duplicate characters (e.g., "HHHELLL" → "HEL")
  4. Map remaining indices to characters using vocabulary
  5. Join into final string

This is the simplest decoding strategy. More advanced strategies like beam search could be added later.

---

### Trainer

#### `trainer/trainer.py`
**Class `Trainer`**
The main training orchestrator.
- `__init__`: Receives model, loaders, criterion, optimizer, device, metrics, logger, checkpoint
- `fit(epochs)`:
  1. Moves model to device
  2. For each epoch:
     - Calls `train_one_epoch` for training
     - Calls `validate` for validation
     - Logs train loss, valid loss, learning rate, predictions, and weight histograms to TensorBoard
     - Saves `last.pt` every epoch
     - Saves `best.pt` when validation loss improves
  3. Closes logger

---

#### `trainer/train_one_epoch.py`
**Function `train_one_epoch(model, dataloader, criterion, optimizer, device, epoch)`**
Runs one complete training epoch:
- Sets `model.train()`
- Iterates over batches with a `tqdm` progress bar
- For each batch: zero gradients → forward → loss → backward → optimizer step
- Tracks and returns average loss for the epoch

---

#### `trainer/validate.py`
**Function `validate(model, dataloader, criterion, device, metrics)`**
Runs validation:
- Sets `model.eval()` and `torch.no_grad()`
- Computes average validation loss
- On the first batch, decodes predictions and stores GT/Pred pairs for logging
- Returns `{"loss": float, "predictions": List[dict]}`

---

#### `trainer/loss.py`
**Class `CTCLossWrapper(nn.Module)`**
Wraps PyTorch's `nn.CTCLoss` with the correct tensor permutations.
- `nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)`
- `forward`: Permutes logits from `(B, T, C)` to `(T, B, C)`, applies `log_softmax`, computes CTC loss

**Why the permutation?** PyTorch's CTC loss expects input in `(T, B, C)` format (time-first), but our model outputs `(B, T, C)` (batch-first).

---

#### `trainer/metrics.py`
**Class `OCRMetrics`**
Evaluates model predictions against ground truth.
- `evaluate_batch(logits, labels, max_samples=5)`: Decodes logits, returns list of `{"gt": str, "pred": str}` for the first `max_samples` in the batch.

---

#### `trainer/logger.py`
**Class `TensorBoardLogger`**
Wraps PyTorch's `SummaryWriter` for TensorBoard logging.
- `log_train_loss(loss, epoch)`, `log_validation_loss(loss, epoch)`, `log_learning_rate(lr, epoch)`
- `log_images(tag, images, epoch)`: Logs image grids
- `log_predictions(gt_texts, pred_texts, epoch, max_samples)`: Logs GT vs predicted text
- `log_model_weights(model, epoch)`: Logs histograms of all parameters
- `log_metrics(results, epoch)`: Logs OCR prediction text under "OCR Predictions" tab
- `log_graph(model, sample_input)`: Logs model computation graph
- `flush()`, `close()`: Standard writer cleanup

---

#### `trainer/checkpoint.py`
**Class `CheckpointManager`**
Manages saving and loading model checkpoints.
- `__init__`: Creates checkpoint directory, initializes `best_loss = inf`
- `save_best(model, optimizer, epoch, loss)`: Saves `best.pt` only if current loss < best_loss
- `save_last(model, optimizer, epoch, loss)`: Always saves `last.pt`
- `load(model, optimizer, path, device)`: Static method to load checkpoint and resume training

Checkpoint contains: epoch, loss, model_state_dict, optimizer_state_dict.

---

### Utilities

#### `utils/logger.py`
Configures Python's standard `logging` module with a consistent format. Used throughout the project for console output.

#### `utils/visualize.py`
**Function `show_batch(batch, count=8)`**
Creates a 2×4 matplotlib figure showing up to 8 images from a batch. Denormalizes pixel values and displays them in grayscale with target indices as titles. Useful for debugging data loading.

---

### Entry Points

#### `main.py`
Integration test / sanity check script. It:
1. Loads vocabulary and dataset
2. Builds each module individually (CNN, Converter, LSTM, Classifier)
3. Assembles the full `CRNN` model
4. Runs forward pass on real data
5. Computes loss with `CTCLossWrapper`
6. Calls `loss.backward()` to verify gradients flow

Run this to quickly verify the entire pipeline works before training.

#### `train.py`
Main training script. It wires everything together:
1. Loads vocabulary
2. Creates train/valid datasets and DataLoaders with `CTCCollate`
3. Instantiates `CRNN`, `CTCLossWrapper`, Adam optimizer
4. Creates `OCRMetrics`, `TensorBoardLogger`, `CheckpointManager`
5. Builds `Trainer` and calls `trainer.fit(epochs=50)`

#### `predict.py`
Currently empty. Will be implemented for standalone inference on single images.

---

## Key Concepts Explained

### CTC Loss — The Secret Sauce

**The Problem:** We want to train a network to map an image to text. But we don't know *which* pixels correspond to *which* characters. Traditional sequence models need aligned inputs and outputs.

**CTC's Solution:** Allow the network to output a sequence longer than the target. Insert a special **blank** token (index 0) between repeated characters during decoding.

Example:
```
Image "HELLO" (5 characters)
Network outputs: [H, E, L, L, <blank>, O, <blank>, <blank>, <blank>, <blank>, ...]
                       ↓ collapse blanks & duplicates
Decoded: "HELLO"
```

CTC loss computes the probability of all possible alignments and sums them. This lets us train end-to-end without character-level bounding boxes.

### Why Bidirectional LSTM?

In scene text, a character may be ambiguous without context from both sides. For example, the character "c" looks similar to "e" or "o". A bidirectional LSTM processes the sequence left-to-right and right-to-left, giving each timestep a richer representation.

### Fixed-Width Input

All images are resized to **32×128**. This makes batching easy (no padding needed for images) and gives a fixed sequence length of 32 for the RNN. Variable-length text is handled by CTC, not by the model architecture.

---

## Training Tips

1. **Start Small**: Use `main.py` first to verify the pipeline. If `loss.backward()` succeeds without errors, your architecture is correct.
2. **Monitor TensorBoard**: Watch train vs. validation loss curves. If train loss drops but valid loss rises, you are overfitting.
3. **Check Predictions**: In TensorBoard's "OCR Predictions" tab, you can see actual GT vs. predicted text after each epoch. This is more meaningful than loss numbers alone.
4. **Best Model**: `checkpoints/best.pt` is automatically saved whenever validation loss improves. Use this for inference.
5. **Character Set**: The vocabulary supports digits (0-9), uppercase (A-Z), and lowercase (a-z) = 62 characters + blank = 63 classes.

---

## Limitations & Future Work

- `models/detector.py` and `preprocessing/image.py` are empty — future work could add text detection (e.g., EAST, DB) to find text regions in full images before recognition
- `predict.py` is not implemented yet — a standalone inference script is needed
- No beam search decoding — greedy decoding is simple but not always optimal
- No data augmentation on text images (rotation, distortion, blur) — could improve robustness

---

## Author

Created as an educational resource for deep learning students progressing from basic CNNs to advanced sequence models.
