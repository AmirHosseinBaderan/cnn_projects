from PIL import Image, ImageDraw, ImageFont


class FaceVisualizer:

    def __init__(self):

        try:
            self.font = ImageFont.truetype(
                "DejaVuSans.ttf",
                20
            )

        except Exception:

            self.font = ImageFont.load_default()

    def draw(
            self,
            image_path,
            results,
            output_path="output.jpg"
    ):

        image = Image.open(
            image_path
        ).convert("RGB")

        draw = ImageDraw.Draw(
            image
        )

        for result in results:

            x1, y1, x2, y2 = result["box"]

            draw.rectangle(
                [(x1, y1), (x2, y2)],
                outline="lime",
                width=3
            )

            text = (
                f'{result["name"]} '
                f'({result["score"]:.2f})'
            )

            draw.text(
                (x1, y1 - 25),
                text,
                fill="lime",
                font=self.font
            )

        image.save(
            output_path
        )

        return output_path