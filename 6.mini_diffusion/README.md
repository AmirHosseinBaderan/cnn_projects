# 6. Mini Diffusion — A Minimal Diffusion Model from Scratch

This project implements a **denoising diffusion probabilistic model (DDPM)** from scratch using PyTorch, trained on the MNIST dataset. The goal is to teach the core mechanics of how diffusion models learn to generate data by gradually removing noise. Every component — from noise scheduling to the U-Net architecture — is built by hand so you can see exactly how a generative model works end to end.

---

## How It Works — The Big Picture

A diffusion model learns in two phases:

1. **Forward process (training only):** Take a clean image and gradually add Gaussian noise over many timesteps until it becomes pure noise. This is fixed — we don't learn it.
2. **Reverse process (training & inference):** Train a neural network to predict the noise that was added at each timestep. At inference time, we start from pure noise and step-by-step remove the predicted noise to recover a clean image.

The model is trained to minimize the difference between the predicted noise and the actual noise that was added.

---

## Project Structure

```
6.mini_diffusion/
├── configs/
│   └── config.py                # Central configuration
├── datasets/
│   ├── mnist.py                 # Basic MNIST wrapper
│   └── diffusion_mnist.py       # Diffusion-specific MNIST dataset
├── diffusion/
│   ├── beta_scheduler.py        # Linear beta schedule
│   ├── noise_scheduler.py       # Adds noise to images
│   ├── forward.py               # (placeholder — forward process)
│   ├── reverse.py               # (placeholder — reverse process)
│   ├── sampler.py               # (placeholder — sampling loop)
│   └── utils.py                 # (placeholder — utilities)
├── models/
│   ├── unet.py                  # U-Net denoising backbone
│   ├── embeddings/
│   │   ├── timestep_embedding.py  # Sinusoidal timestep encoding
│   │   └── label_embedding.py     # Class label encoding
│   └── blocks/
│       ├── conv_block.py        # Conv → BN → ReLU building block
│       ├── down_block.py        # Encoder down-sampling block
│       ├── up_block.py          # Decoder up-sampling block
│       ├── bottleneck.py        # Bottleneck with conditional injection
│       ├── conditional.py       # Linear projection for conditioning
│       ├── residual.py          # (placeholder)
│       ├── attention.py         # (placeholder)
│       ├── down.py              # (placeholder)
│       └── up.py                # (placeholder)
├── trainer/
│   ├── trainer.py               # Training loop logic
│   └── loss.py                  # (placeholder — loss utilities)
├── inference/
    │   ├── generator.py       # (placeholder — generation script)
    │   ├── predictor.py       # Prediction entry point
    │   └── diffusion_predictor.py  # Full diffusion prediction pipeline
├── utils/
│   ├── logger.py                # Logging setup
│   └── checkpoint_manager.py    # Save/load model checkpoints
├── train.py                     # Entry point for training
├── predict.py                   # (placeholder — inference entry point)
└── tests/
    └── test_forward.py          # Visualize noise addition at various timesteps
```

---

## Classes — Explained

### `Config` (`configs/config.py`)

**What it does:** This is the central configuration class that holds all hyperparameters and settings for the project in one place. Instead of scattering magic numbers throughout the code, `Config` gives every component a single source of truth.

**How it works:** `Config` is a plain Python class with class-level attributes. It automatically detects whether a CUDA-capable GPU is available and sets `DEVICE` accordingly. All other values — image size, number of classes, batch size, number of training epochs, and early-stopping patience — are defined as simple constants.

**Key attributes:**
- `DEVICE` — `"cuda"` if a GPU is available, otherwise `"cpu"`
- `IMAGE_SIZE` — `28` (MNIST images are 28×28 pixels)
- `NUM_CLASSES` — `10` (digits 0–9)
- `BATCH_SIZE` — `64` samples per training step
- `EPOCHS` — `10` total training epochs
- `PATIENCE` — `3` epochs without improvement before early stopping kicks in
- `DATA_DIR` — path to the data directory

---

### `UNet` (`models/unet.py`)

**What it does:** The U-Net is the neural network that learns to predict the noise added to a noisy image. It is the heart of the diffusion model — everything else exists to feed data into and out of this network.

**How it works:** The U-Net gets three inputs: a noisy image, a timestep (telling the network *how much* noise was added), and a class label (telling the network *which digit* to generate). It produces a single output: the predicted noise.

The architecture follows the classic U-Net shape — an **encoder** that shrinks the spatial dimensions while growing the channel depth, a **bottleneck** that processes the most abstract representation, and a **decoder** that grows the spatial dimensions back up while shrinking the channels. Skip connections from the encoder are concatenated with the decoder at each level, allowing the network to retain fine spatial details that would otherwise be lost during down-sampling.

The timestep and label embeddings are combined (added together) and injected into the bottleneck via a conditional projection, so the network learns to denoise differently depending on both *when* in the noise schedule and *which class* it is dealing with.

**Architecture flow:**
1. `encoder` (ConvBlock): 1 → 32 channels, spatial size preserved
2. `down` (DownBlock): 32 → 64 channels, spatial size halved
3. `bottleneck` (Bottleneck): 64 → 128 channels, receives time+label conditioning
4. `up` (UpBlock): 128 → 64 → 32 channels, spatial size doubled with skip connections
5. `output` (1×1 Conv): 32 → 1 channel, produces the predicted noise

---

### `TimeEmbedding` (`models/embeddings/timestep_embedding.py`)

**What it does:** Converts a discrete timestep integer (e.g., `t = 500`) into a continuous vector that the neural network can use. This is essential because the network needs to know *how much noise* was added to the image so it can learn to remove the right amount.

**How it works:** This class implements **sinusoidal positional encoding**, the same technique used in the Transformer architecture. Given a timestep `t`, it computes:

```
embeddings = [sin(t / 10000^(2i/d)), cos(t / 10000^(2i/d))] for i in 0..d/2-1
```

where `d` is the embedding dimension. The `log(10000)` scaling ensures that different timesteps produce widely separated vectors, giving the network a strong signal about the noise level. The result is a vector of size `embedding_dim` that is added to the label embedding to form the conditioning signal.

**Why sinusoidal?** Unlike learned embeddings, sinusoidal encodings generalize to timesteps the model has never seen during training (e.g., if the model was trained on timesteps 0–999 but inference uses a different schedule). They also provide a smooth, continuous representation where nearby timesteps have similar vectors.

---

### `LabelEmbedding` (`models/embeddings/label_embedding.py`)

**What it does:** Converts a class label (an integer 0–9 for MNIST) into a dense vector embedding. This allows the U-Net to condition its denoising on the digit class — so it learns to remove noise in a way that is aware of *which digit* the image should be.

**How it works:** `LabelEmbedding` is a simple `nn.Module` that contains an `nn.Embedding` layer. The embedding layer is a lookup table of shape `(num_classes, embedding_dim)` — for MNIST, that is `(10, 64)`. When given a label tensor (e.g., `[3]`), it returns the corresponding row from the table as a dense vector.

The label embedding is added to the time embedding before being passed to the bottleneck, so the network receives a combined conditioning signal that encodes both *when* (timestep) and *what* (class).

---

### `ConvBlock` (`models/blocks/conv_block.py`)

**What it does:** A reusable building block that applies two convolutional layers with batch normalization and ReLU activation between them. It is the fundamental unit used throughout the U-Net encoder, decoder, and bottleneck.

**How it works:** The block consists of a `nn.Sequential` pipeline:
1. `Conv2d` (3×3, padding=1) — preserves spatial dimensions
2. `BatchNorm2d` — normalizes activations for stable training
3. `ReLU` — introduces non-linearity
4. `Conv2d` (3×3, padding=1) — preserves spatial dimensions again
5. `BatchNorm2d` — normalizes again
6. `ReLU` — final non-linearity

The number of input and output channels is configurable, allowing the block to be used at different depths of the U-Net where channel counts differ.

---

### `DownBlock` (`models/blocks/down_block.py`)

**What it does:** The encoder block of the U-Net. It processes features through a `ConvBlock` and then down-samples the spatial dimensions by a factor of 2, while the original (pre-downsample) features are returned as a **skip connection**.

**How it works:** In the forward pass:
1. The input passes through a `ConvBlock` to produce processed features.
2. The processed features are saved as `skip` — these will be concatenated later in the decoder.
3. A `Conv2d` with `stride=2` and `kernel_size=4` down-samples the spatial dimensions by half.
4. Both the down-sampled features and the skip connection are returned.

Skip connections are critical in U-Nets because they allow the decoder to recover fine spatial details that are lost during down-sampling. Without them, the network would struggle to produce sharp outputs.

---

### `UpBlock` (`models/blocks/up_block.py`)

**What it does:** The decoder block of the U-Net. It up-samples the spatial dimensions and fuses features from the corresponding encoder skip connection, then processes the combined features through a `ConvBlock`.

**How it works:** In the forward pass:
1. The input is up-sampled by a factor of 2 using `ConvTranspose2d` (kernel_size=2, stride=2).
2. The up-sampled features are concatenated channel-wise with the `skip` tensor from the encoder. This is where the skip connection is consumed.
3. The concatenated tensor passes through a `ConvBlock` to produce the output features.

The concatenation (not addition) of skip connections preserves all spatial information from the encoder, giving the decoder the best possible signal to reconstruct fine details.

---

### `Bottleneck` (`models/blocks/bottleneck.py`)

**What it does:** The deepest layer of the U-Net where the spatial dimensions are smallest and the channel dimension is largest. It is here that the timestep and label conditioning are injected into the feature representation.

**How it works:** In the forward pass:
1. The conditioning vector (combined time + label embedding) is projected from `condition_dim` to `channels` using a linear layer.
2. The projected vector is reshaped to `(batch, channels, 1, 1)` so it can be broadcast-added to the feature map.
3. The condition is added to the input features — this is **conditional injection** (also called FiLM-like conditioning). It shifts and scales the features based on the timestep and class.
4. The conditioned features pass through a `ConvBlock` that expands channels from `channels` to `channels * 2`.

This design means the network learns different denoising behaviors for different timesteps and classes, all controlled by a simple additive conditioning signal.

---

### `ConditionProjection` (`models/blocks/conditional.py`)

**What it does:** A standalone linear projection layer that maps a conditioning vector (like the combined time+label embedding) to a feature-space dimension compatible with the bottleneck's channel count.

**How it works:** It wraps a single `nn.Linear(embedding_dim, channels)` layer. In the `Bottleneck`, this projection is applied to the combined condition vector before it is reshaped and added to the feature map.

While the `Bottleneck` already uses this projection internally, `ConditionProjection` is provided as a reusable standalone module in case you want to inject conditioning at other points in the network (e.g., at every U-Net level instead of just the bottleneck).

---

### `NoiseScheduler` (`diffusion/noise_scheduler.py`)

**What it does:** Implements the forward diffusion process — the fixed procedure for adding Gaussian noise to a clean image at any given timestep `t`. This is used during training to create noisy images that the U-Net learns to denoise.

**How it works:** The scheduler precomputes three sequences during initialization:
1. **`betas`** — a linearly spaced sequence from `1e-4` to `2e-2` over `num_timesteps` (default 1000). These control how much noise is added at each step.
2. **`alphas`** — `1 - betas`, representing the signal retention at each step.
3. **`alpha_bars`** — the cumulative product of `alphas`. `alpha_bar[t]` represents the total signal retained after `t` steps of noise addition.

When `add_noise(images, timesteps)` is called:
1. Random Gaussian noise is generated (same shape as the images).
2. `alpha_bar[t]` is looked up for each timestep in the batch.
3. The noisy image is computed as: `sqrt(alpha_bar) * image + sqrt(1 - alpha_bar) * noise`.

At `t=0`, `alpha_bar ≈ 1`, so the image is nearly unchanged. At `t=999`, `alpha_bar ≈ 0`, so the image is almost pure noise. The model is trained to predict the `noise` given the `noisy_image` and the timestep `t`.

---

### `linear_beta_schedule` (`diffusion/beta_scheduler.py`)

**What it does:** Generates a linearly spaced schedule of beta values that control the noise addition rate across timesteps.

**How it works:** It simply calls `torch.linspace(beta_start, beta_end, num_timesteps)` to produce a 1D tensor of `num_timesteps` values evenly spaced between `beta_start` (default `1e-4`) and `beta_end` (default `2e-2`).

The linear schedule is the simplest choice — it adds a small, constant amount of noise at each step. More advanced schedules (cosine, quadratic) can produce better results but are not implemented yet in this project.

---

### `DiffusionMNISTDataset` (`datasets/diffusion_mnist.py`)

**What it does:** A PyTorch `Dataset` wrapper around the MNIST dataset that returns samples in a dictionary format suitable for the diffusion training loop.

**How it works:** It inherits from `torch.utils.data.Dataset` and internally uses `torchvision.datasets.MNIST` with `ToTensor()` transform. The `__getitem__` method returns a dictionary with two keys:
- `"image"` — the MNIST image tensor (shape `[1, 28, 28]`)
- `"label"` — the digit label as a `torch.tensor` (scalar)

The dictionary format is important because the `Trainer._step` method accesses `batch["image"]` and `batch["label"]` directly.

---

### `MNISTDataset` (`datasets/mnist.py`)

**What it does:** A simpler MNIST wrapper that returns raw `(image, label)` tuples instead of dictionaries. This is a basic utility class; the diffusion training uses `DiffusionMNISTDataset` instead.

**How it works:** It wraps `torchvision.datasets.MNIST` and returns `(image, label)` from `__getitem__`. The `__len__` delegates to the underlying dataset.

---

### `Trainer` (`trainer/trainer.py`)

**What it does:** Encapsulates the training and validation logic for the diffusion model. It handles the complete training step: sampling random timesteps, adding noise, running the U-Net forward pass, and computing the loss.

**How it works:**

The core `_step` method:
1. Moves the batch images and labels to the configured device.
2. Samples random timesteps uniformly from `[0, num_timesteps)` for each item in the batch.
3. Calls `scheduler.add_noise(images, timesteps)` to produce noisy images and captures the ground-truth noise.
4. Passes `(noisy_images, timesteps, labels)` through the U-Net to get `predicted_noise`.
5. Computes the loss as `MSELoss(predicted_noise, noise)` — the mean squared error between the predicted and actual noise.

The `train_step` method wraps `_step` with backpropagation: zero gradients, backward pass, optimizer step. The `validation_step` method wraps `_step` in `torch.no_grad()` for evaluation without gradient computation.

**Why MSE?** In DDPM, the loss is simply the mean squared error between the predicted noise and the actual noise. This is equivalent to minimizing the variational lower bound on the log-likelihood, which is the theoretical foundation of diffusion models.

---

### `CheckpointManager` (`utils/checkpoint_manager.py`)

**What it does:** Handles saving and loading model checkpoints during training. It saves the last checkpoint every epoch and the best checkpoint whenever a new lowest validation loss is achieved.

**How it works:**

- `save_checkpoint(epoch, val_loss, is_best, best_val_loss)` creates a dictionary containing the epoch, model state dict, optimizer state dict, and validation loss. It always saves to `last_checkpoint.pt` and additionally saves to `best_checkpoint.pt` if `is_best` is true.
- `load_checkpoint(checkpoint_path)` loads a checkpoint file and restores the model, optimizer, and (optionally) scheduler state. It returns the epoch, validation loss, and best validation loss so training can resume from the correct state.

This enables **training resumption** — if training is interrupted, you can reload the last checkpoint and continue from where you left off.

---

### `logger` (`utils/logger.py`)

**What it does:** Provides a configured Python logger for the project. It is a singleton module-level logger, not a class.

**How it works:** The module configures the root logging system with `INFO` level and a format showing timestamp, log level, and message. It then creates a named logger via `logging.getLogger(__name__)`. All modules in the project use this shared logger to print training progress and status messages.

---

## Training Flow (`train.py`)

The training script ties everything together:

1. **Data:** Creates a `DiffusionMNISTDataset`, splits it 90/10 into train/validation, and wraps each in a `DataLoader`.
2. **Model & Scheduler:** Instantiates `UNet` and `NoiseScheduler`, and moves the model to the configured device.
3. **Optimizer:** Uses Adam with learning rate `1e-4`.
4. **Trainer:** Wraps the model, scheduler, optimizer, and device into a `Trainer` instance.
5. **Checkpointing:** Creates a `CheckpointManager` and attempts to load a previous checkpoint to resume training.
6. **Loop:** For each epoch, runs `trainer.train_step()` over all training batches and `trainer.validation_step()` over all validation batches. Logs average losses, saves checkpoints, and triggers early stopping if validation loss doesn't improve for `PATIENCE` epochs.

---

## Testing (`tests/test_forward.py`)

The test script visualizes the noise addition process at different timesteps. It loads a single MNIST image, adds noise at timesteps `t = 0, 100, 300, 500, 700, 999`, and displays the results in a row of subplots. This is a great way to verify that the forward process works correctly — at `t=0` the image should be clean, and at `t=999` it should look like pure noise.

---

## Key Concepts to Understand

| Concept | What it means |
|---------|---------------|
| **Timestep (t)** | An integer from 0 to 999 indicating how many noise-adding steps have been applied. t=0 is clean, t=999 is pure noise. |
| **Alpha bar (α̅)** | The cumulative product of alphas up to timestep t. It represents how much of the original signal remains. |
| **Noise prediction** | The U-Net's task: given a noisy image and timestep t, predict the Gaussian noise that was added. |
| **Skip connections** | Direct paths from encoder to decoder that preserve spatial detail. They are concatenated, not added. |
| **Conditional generation** | By feeding class labels into the network, we can control *which digit* the model generates at inference time. |
| **MSE loss** | The training objective — minimize the difference between predicted noise and actual noise. |

---

## Running the Project

```bash
# From the project root
cd 6.mini_diffusion

# Train the model
python train.py

# Visualize noise addition
python tests/test_forward.py
```

## Prerequisites

- Python 3.8+
- PyTorch 2.0+
- torchvision
- matplotlib (for test visualization)
- tqdm (for progress bars)