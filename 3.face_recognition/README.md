# Project 3: Face Recognition System

## What is this project?

This project teaches you how to build a **real-time face recognition system** that can:
1. **Detect faces** in images and video streams
2. **Recognize known people** by comparing their faces to a database
3. **Register new people** into the system
4. **Work in real-time** through a webcam

Unlike Projects 1 and 2 which use standard classification (predicting from fixed categories), this project uses **metric learning** - a more advanced technique where the model learns to measure similarity between faces.

## What will you learn?

By completing this project, you will understand:

1. **Metric Learning** - Learning similarity instead of just classification
2. **Triplet Loss** - A special loss function for learning embeddings
3. **Face Detection** - Using MTCNN to find faces in images
4. **Face Embeddings** - Converting faces to numerical vectors
5. **Cosine Similarity** - Measuring how similar two faces are
6. **Real-time Applications** - Processing video streams with OpenCV

## Project Structure

```
3.face_recognition/
├── main.py                  # GUI application (Tkinter)
├── train.py                 # Train with triplet loss
├── webcam.py                # Real-time face recognition
├── register_cam.py          # Register faces via webcam
├── recognize_image.py       # Recognize faces in images
├── evaluate.py              # Evaluate model accuracy
├── search_face.py           # Search for a face in database
├── register_face.py         # Register face from folder
├── src/
│   ├── model.py             # FaceEmbeddingNet architecture
│   ├── loss.py              # TripletLoss implementation
│   ├── detector.py          # MTCNN face detection
│   ├── database.py          # Face database (save/load embeddings)
│   ├── search.py            # Find similar faces
│   ├── recognizer.py        # Combine detection + recognition
│   ├── evaluator.py         # Get face embeddings
│   ├── visualizer.py        # Draw results on images
│   ├── triplet_dataset.py   # Generate triplets for training
│   ├── config.py            # Configuration
│   ├── checkpoint.py        # Save/load model
│   ├── calibrator.py        # Find optimal threshold
│   └── transforms.py        # Face preprocessing
├── data/
│   ├── raw/                 # Training images (organized by person)
│   └── database/            # Saved face embeddings
└── checkpoints/             # Trained model weights
```

## How It Works - The Big Picture

### 1. The Problem with Standard Classification

In Projects 1 and 2, we used **softmax classification**:
- Input → Model → Output (scores for each class)
- We had a fixed number of classes (10 or 6)
- We couldn't add new classes without retraining

For face recognition, this doesn't work because:
- We don't know all people in advance
- New people need to be added dynamically
- We want to recognize faces, not just classify them

### 2. The Solution: Metric Learning

Instead of classifying, we learn to **measure similarity**:

```
Face A → [Model] → Embedding Vector (128 numbers)
Face B → [Model] → Embedding Vector (128 numbers)

Similarity = CosineSimilarity(Embedding A, Embedding B)
```

**The key idea**: If two faces belong to the same person, their embeddings should be similar (close together). If they're different people, their embeddings should be different (far apart).

### 3. Triplet Loss - The Training Strategy

To learn these embeddings, we use **Triplet Loss**. The idea is simple:

```
For each training step, we take 3 images:
- Anchor: A face of person X
- Positive: Another face of person X (same person)
- Negative: A face of person Y (different person)

We want: distance(Anchor, Positive) < distance(Anchor, Negative)
```

```python
# From src/loss.py
class TripletLoss(nn.Module):
    def forward(self, anchor, positive, negative):
        positive_distance = pairwise_distance(anchor, positive)
        negative_distance = pairwise_distance(anchor, negative)
        
        # We want positive_distance to be smaller than negative_distance
        loss = relu(positive_distance - negative_distance + margin)
        return loss.mean()
```

**The margin** is a safety buffer. We don't just want the positive to be closer - we want it to be noticeably closer.

### 4. The Face Embedding Network

Our model (`src/model.py`) converts a face image into a 128-dimensional vector:

```
Input Face (3, 112, 112)
    ↓
[Conv2d: 3→32] → BatchNorm → ReLU → MaxPool
    ↓
[Conv2d: 32→64] → BatchNorm → ReLU → MaxPool
    ↓
[Conv2d: 64→128] → BatchNorm → ReLU → AdaptiveAvgPool
    ↓
Flatten → [Linear: 128→128]
    ↓
L2 Normalize → Output (128-dim embedding)
```

**Why L2 Normalize?** It scales the embedding vector to have length 1, making cosine similarity calculations more meaningful.

### 5. Face Detection with MTCNN

Before we can recognize a face, we need to find it in the image:

```python
# From src/detector.py
class FaceDetector:
    def __init__(self, device):
        self.detector = MTCNN(
            image_size=112,
            keep_all=True,
            device=device
        )
    
    def detect(self, image):
        boxes, probabilities = self.detector.detect(image)
        # Returns list of face locations and confidence scores
```

**MTCNN** (Multi-task Cascaded Convolutional Networks) is a pre-trained model that:
1. Finds all faces in an image
2. Aligns them (rotates to be upright)
3. Crops them to a standard size (112x112)

### 6. The Recognition Pipeline

When we want to recognize a face:

```python
# 1. Detect faces in the image
detections = detector.detect(frame)

# 2. For each detected face
for detection in detections:
    # 3. Get the face embedding
    embedding = model(detection["face"])
    
    # 4. Search the database for similar faces
    result = search_engine.search(embedding)
    
    # 5. Return the name and confidence
    results.append({
        "name": result["name"],      # "George_W_Bush" or "Unknown"
        "score": result["score"],    # Similarity score (0-1)
        "box": detection["box"]      # Where the face is in the image
    })
```

### 7. The Database System

We store face embeddings in a simple file-based database:

```python
# From src/database.py
class FaceDatabase:
    def add_face(self, person_name, embedding):
        database = self.load()
        if person_name not in database:
            database[person_name] = []
        database[person_name].append(embedding)
        self.save(database)
```

**Why store multiple embeddings per person?** A person's face looks different under different lighting, angles, and expressions. Storing multiple embeddings makes recognition more robust.

### 8. Similarity Search

To find who a face belongs to:

```python
# From src/search.py
def search(self, query_embedding, threshold=0.7):
    best_person = None
    best_score = -1.0
    
    for person, embeddings in self.database.items():
        for embedding in embeddings:
            score = cosine_similarity(query_embedding, embedding)
            if score > best_score:
                best_score = score
                best_person = person
    
    if best_score < threshold:
        return {"name": "Unknown", "score": best_score}
    return {"name": best_person, "score": best_score}
```

**The threshold** (0.7) is calibrated during evaluation. If the best match is below this score, we say "Unknown" rather than guessing wrong.

## How to Run

### Step 1: Install Dependencies

```bash
pip install torch torchvision opencv-python pillow facenet-pytorch scikit-learn
```

### Step 2: Train the Model

```bash
cd 3.face_recognition
python train.py
```

This will:
1. Load face images from `data/raw/` (organized by person name)
2. Generate triplets (anchor, positive, negative)
3. Train for 20 epochs with early stopping
4. Save the best model to `checkpoints/face_embedding.pth`

**Data format:**
```
data/raw/
├── George_W_Bush/
│   ├── George_W_Bush_0001.jpg
│   ├── George_W_Bush_0002.jpg
│   └── ...
├── Tony_Blair/
│   └── ...
└── ...
```

### Step 3: Evaluate the Model

```bash
python evaluate.py
```

This finds the optimal threshold by testing same-person and different-person pairs.

### Step 4: Register Faces

```bash
python register_cam.py --name "John_Doe"
```

This opens your webcam and captures 10 photos of the person, storing their face embedding.

### Step 5: Run Real-time Recognition

```bash
python webcam.py
```

This opens your webcam and recognizes faces in real-time, displaying names and confidence scores.

### Step 6: Use the GUI

```bash
python main.py
```

A simple GUI with buttons for:
- **Register**: Add a new person to the database
- **Capture**: Start real-time recognition
- **Exit**: Close the application

## Key Concepts Explained

### What is an Embedding?

An embedding is a compact numerical representation of something. For faces:
- Input: 112x112 pixel image (12,544 numbers)
- Output: 128-dimensional vector (128 numbers)

The model compresses the image into a vector where similar faces have similar vectors.

### What is Triplet Loss?

Triplet loss trains the model using three images at a time:
- **Anchor**: The reference image
- **Positive**: Same person as anchor
- **Negative**: Different person from anchor

The loss function pushes the anchor and positive closer together while pulling the anchor and negative farther apart.

### What is Cosine Similarity?

A measure of how similar two vectors are, ranging from -1 (opposite) to 1 (identical). For normalized face embeddings, it's simply the dot product:

```
similarity = embedding_A · embedding_B
```

### What is MTCNN?

MTCNN (Multi-task Cascaded Convolutional Networks) is a pre-trained deep learning model that:
1. Detects faces in images
2. Aligns them (corrects rotation)
3. Crops them to a standard size

It's more accurate than Haar cascades and works well with different face orientations.

## Expected Results

With sufficient training data (50+ images per person), you should achieve:
- **95%+ accuracy** on known faces
- **Good generalization** to new photos of registered people
- **Real-time performance** (30+ FPS on GPU)

## Troubleshooting

- **MTCNN not detecting faces?** Ensure good lighting and frontal face orientation
- **Low recognition accuracy?** Register more images per person (at least 10-20)
- **Too many "Unknown" predictions?** Lower the threshold in `src/search.py`
- **CUDA out of memory?** Reduce `BATCH_SIZE` in `src/config.py`
- **Training too slow?** Use a GPU or reduce the dataset size

## Next Steps

After completing this project, you can extend it with:
- Face anti-spoofing (prevent photo attacks)
- Age and gender estimation
- Emotion recognition
- Integration with attendance systems
- Mobile deployment
