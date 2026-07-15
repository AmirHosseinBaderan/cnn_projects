import cv2
import numpy as np
from utils.image import load_image,save_image,show_image
from preprocessing.grayscale import to_grayscale
from preprocessing.pipeline import PreprocessingPipeline

path = "samples/noise-image.png"

image = load_image(path)
print(image.shape)
print(image.dtype)

#show_image(image)

# for i in range(100):
#     image[i, i] = [255, 0, 0]
#     for j in range(i):
#         image[i, j] = [255, 0, 0]
#
# save_image("samples/img-test-2.jpg",image=image)

print(image.min())
print(image.max())
print(image.mean())


pipeline = PreprocessingPipeline()
result = pipeline.process(image)

save_image("samples/noise-image-2.png",image=result)

show_image(result)