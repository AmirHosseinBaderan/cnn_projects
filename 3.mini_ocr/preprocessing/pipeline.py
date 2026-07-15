from .grayscale import to_grayscale
from .threshold import otsu_threshold
from .blur import median_blur
from .morphology import opening, closing


class PreprocessingPipeline:

    def process(self, image):
        image = to_grayscale(image)
        image = median_blur(image)
        image = otsu_threshold(image)

        image = opening(image)
        image = closing(image)

        return image
