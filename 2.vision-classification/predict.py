import torch
from models.simple_cnn import SimpleCNN
from PIL import Image
from transforms.transforms import test_transform

MODEL_PATH = "checkpoints/model.pth"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = SimpleCNN().to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

image = Image.open("data/raw/seg_pred/3.jpg").convert("RGB")
image = test_transform(image)
image = image.unsqueeze(0)

with torch.no_grad():
    outputs = model(image.to(device))

probabilities = torch.softmax(
    outputs,
    dim=1
)
confidence = probabilities.max().item()
predicted = probabilities.argmax(dim=1)

CLASS_NAMES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]

prediction = CLASS_NAMES[
    predicted.item()
]
print(
    f"{prediction}: {confidence:.2%}"
)