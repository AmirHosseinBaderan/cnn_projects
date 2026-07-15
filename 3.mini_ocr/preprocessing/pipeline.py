from .grayscale import to_grayscale
from .threshold import otsu_threshold
from .blur import median_blur

class PreprocessingPipeline:

    def process(self, image):
        image = to_grayscale(image)
        image = median_blur(image)
        image = otsu_threshold(image)

        return image
