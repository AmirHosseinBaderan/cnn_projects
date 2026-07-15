from .grayscale import to_grayscale
from .threshold import otsu_threshold


class PreprocessingPipeline:

    def process(self, image):
        image = to_grayscale(image)
        image = otsu_threshold(image)

        return image
