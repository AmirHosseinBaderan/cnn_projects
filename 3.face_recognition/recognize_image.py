from src.config import Config
from src.database import FaceDatabase
from src.detector import FaceDetector
from src.model import FaceEmbeddingNet
from src.search import FaceSearch
from src.recognizer import FaceRecognizer
from src.checkpoint import load_checkpoint
from src.visualizer import FaceVisualizer


database = FaceDatabase()

data = database.load()

search = FaceSearch(data)

model = FaceEmbeddingNet().to(
    Config.DEVICE
)

load_checkpoint(
    Config.CHECKPOINT_PATH,
    model
)

model.eval()


detector = FaceDetector(
    Config.DEVICE
)


recognizer = FaceRecognizer(
    detector=detector,
    model=model,
    search_engine=search,
    device=Config.DEVICE
)


results = recognizer.recognize(
    "./test/group.jpg"
)

for result in results:
    print(result)

visualizer = FaceVisualizer()

output = visualizer.draw(
    "./test/group.jpg",
    results
)

print(f"Saved : {output}")