from inference.engine import OCREngine

ocr = OCREngine(
    checkpoint_path="checkpoints/best.pt",
    vocabulary_file="resources/vocab.json",
)

text = ocr.predict("data/test/images.png")

print(text)