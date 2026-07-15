import cv2
import numpy as np
from utils.image import load_image, show_image
from preprocessing.pipeline import PreprocessingPipeline
from detection.contour_detector import ContourDetector

contour_detector = ContourDetector()


def bounding_box():
    output = result.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (255, 255, 255),
            1
        )

    show_image(output)


path = "samples/noise-image.png"

image = load_image(path)
# show_image(image)

# for i in range(100):
#     image[i, i] = [255, 0, 0]
#     for j in range(i):
#         image[i, j] = [255, 0, 0]
#
# save_image("samples/img-test-2.jpg",image=image)

# print(image.min())
# print(image.max())
# print(image.mean())

pipeline = PreprocessingPipeline()
result = pipeline.process(image)

contours = contour_detector.detect(result)
print(f'Contours detected: {len(contours)}')

bounding_box()
