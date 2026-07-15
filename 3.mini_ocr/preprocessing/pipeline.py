from .grayscale import to_grayscale

class PreprocessingPipeline:

    def process(self, image):
        image = to_grayscale(image)
        return image