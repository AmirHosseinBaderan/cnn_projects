from PIL import Image


class ResizeHeight:
    def __init__(
            self,
            height: int
    ):
        self.height = height

    def __call__(
            self,
            image:Image.Image
    ):
        width, height = image.size
        new_width = int(width * self.height / height)
        image = image.resize(
            (new_width, self.height),
            Image.BILINEAR
        )

        return image
