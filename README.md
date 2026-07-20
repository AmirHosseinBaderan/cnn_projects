# CNN Projects

A comprehensive collection of four deep learning projects focused on computer vision and convolutional neural networks (CNNs). These projects are designed as educational resources for students learning about deep learning, computer vision, and PyTorch.

## Projects Overview

This repository contains three progressively advanced projects:

| # | Project | Description | Key Concepts |
| --- |---------|-------------|--------------|
| 1 | [Image Classification (CIFAR-10)](1.cnn/README.md) | Multi-class image classification using custom CNN, ResNet, and EfficientNet | CNN architecture, transfer learning, data augmentation |
| 2 | [Scene Classification](2.vision-classification/README.md) | Natural scene classification (buildings, forest, glacier, mountain, sea, street) | Custom CNN design, early stopping, learning rate scheduling |
| 3 | [Face Recognition](3.face_recognition/README.md) | Real-time face recognition system with webcam support | Triplet loss, face embeddings, MTCNN detection, cosine similarity |
| 4 | [OCR — Scene Text Recognition](4.ocr/README.md) | End-to-end text recognition from images using CRNN and CTC loss | CRNN, CTC loss, BiLSTM, sequence modeling, greedy decoding |

## Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended for training)
- Basic understanding of deep learning and Python

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install dependencies
pip install torch torchvision opencv-python pillow scikit-learn matplotlib facenet-pytorch
```

## Project Structure

```
cnn_projects/
├── 1.cnn/                    # CIFAR-10 Image Classification
│   ├── config.py            # Configuration settings
│   ├── train.py             # Training script
│   ├── predict.py           # Prediction script
│   ├── models/              # Model architectures
│   ├── datasets/            # Data loading and transforms
│   ├── engine/              # Training and evaluation loops
│   └── utils/               # Helper utilities
│
├── 2.vision-classification/ # Intel Scene Classification
│   ├── train.py             # Training script
│   ├── predict.py           # Prediction script
│   ├── evaluator.py         # Evaluation logic
│   ├── datasets/            # Custom dataset classes
│   ├── models/              # Model definitions
│   ├── transforms/          # Image transformations
│   └── utils/               # Utilities
│
└── 3.face_recognition/      # Face Recognition System
    ├── main.py              # GUI application
    ├── train.py             # Triplet loss training
    ├── webcam.py            # Real-time recognition
    ├── register_cam.py      # Face registration via camera
    ├── recognize_image.py   # Image-based recognition
    ├── evaluate.py          # Model evaluation
    └── src/                 # Core modules
        ├── model.py         # Face embedding network
        ├── loss.py          # Triplet loss
        ├── detector.py      # MTCNN face detection
        ├── database.py      # Face database management
        ├── search.py        # Similarity search
        └── ...

├── 4.ocr/                    # Scene Text Recognition (CRNN + CTC)
    ├── config.py            # Hyperparameters
    ├── main.py              # Architecture sanity check
    ├── train.py             # Training script
    ├── predict.py           # Inference placeholder
    ├── models/              # CRNN architecture
    │   ├── recognizer.py    # Full CRNN model
    │   └── modules/         # Sub-modules
    │       ├── cnn.py       # CNN feature extractor
    │       ├── sequence.py  # BiLSTM sequence model
    │       └── classifier.py # CTC classifier
    ├── dataset/             # Data loading & vocabulary
    │   ├── synth90k.py      # Image+text dataset
    │   ├── vocabulary.py    # Character encoding
    │   ├── collate.py       # CTC collate function
    │   ├── build_vocab.py   # Vocabulary builder
    │   └── split.py         # Train/valid split
    ├── preprocessing/       # Image transforms
    │   └── transforms.py    # Resize + normalize
    ├── decoder/             # Greedy decoder
    │   └── greedy.py        # Argmax + blank removal
    ├── trainer/             # Training loop & utilities
    │   ├── trainer.py       # Main trainer orchestrator
    │   ├── train_one_epoch.py # Single epoch loop
    │   ├── validate.py      # Validation loop
    │   ├── loss.py          # CTC loss wrapper
    │   ├── metrics.py       # OCR metrics
    │   ├── logger.py        # TensorBoard logger
    │   └── checkpoint.py    # Checkpoint manager
    └── utils/               # Helpers
        ├── logger.py        # Logging config
        └── visualize.py     # Batch visualization
```

## Learning Path

We recommend completing the projects in order:

1. **Start with Project 1** - Learn the fundamentals of CNN training, data loading, and evaluation
2. **Move to Project 2** - Understand custom dataset handling and advanced training techniques
3. **Continue to Project 3** - Explore metric learning and real-time applications
4. **Finish with Project 4** - Dive into sequence models, CTC loss, and end-to-end OCR

Each project includes detailed documentation explaining the concepts, implementation details, and how to run the code.

## Author

Created as an educational resource for deep learning students.
