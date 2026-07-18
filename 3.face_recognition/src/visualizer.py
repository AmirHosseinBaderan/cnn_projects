import cv2
import numpy as np

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
        ).convert(
            "RGB"
        )

        draw = ImageDraw.Draw(
            image
        )

        for result in results:

            x1, y1, x2, y2 = result["box"]

            color = "lime"

            if result["name"] == "Unknown":
                color = "red"

            draw.rectangle(
                [(x1, y1), (x2, y2)],
                outline=color,
                width=3
            )

            text = (
                f'{result["name"]} '
                f'({result["score"]:.2f})'
            )

            draw.text(
                (x1, y1 - 25),
                text,
                fill=color,
                font=self.font
            )

        image.save(
            output_path
        )

        return output_path

    def draw_frame(
            self,
            frame,
            results
    ):

        output = frame.copy()

        for result in results:

            x1, y1, x2, y2 = result["box"]

            if result["name"] == "Unknown":
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            text = (
                f'{result["name"]} '
                f'{result["score"]:.2f}'
            )

            cv2.putText(
                output,
                text,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA
            )

        return output