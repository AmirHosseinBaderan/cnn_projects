import cv2
import numpy as np
from utils.image import load_image, show_image
from preprocessing.pipeline import PreprocessingPipeline
from detection.contour_detector import ContourDetector
from detection.connected_components import ConnectedComponentDetector

contour_detector = ContourDetector()
connected_component = ConnectedComponentDetector()


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


def connected_bounding_box():
    output = result.copy()

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

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


num_labels, labels, stats, centroids = connected_component.detect(result)
connected_bounding_box()

contours = contour_detector.detect(result)
bounding_box()
