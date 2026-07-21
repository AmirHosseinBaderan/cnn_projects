import cv2

from inference.engine import OCREngine


ocr = OCREngine(
    checkpoint_path="checkpoints/best.pt",
    vocabulary_file="resources/vocab.json",
)

image = cv2.imread("data/test/test_6.png")

text = ocr.predict(image)

print(text)