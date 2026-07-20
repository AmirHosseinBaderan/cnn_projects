import cv2

class LineDetector:
    def __init__(
        self,
        min_height=10,
        min_width=30,
        kernel_width=40,
        kernel_height=3,
    ):
        self.min_height = min_height
        self.min_width = min_width
        self.kernel_width = kernel_width
        self.kernel_height = kernel_height

    def detect(self, image):
        # image:
        # numpy.ndarray (Gray or BGR)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )
        else:
            gray = image.copy()

        # Binary Image
        binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )[1]

        # Connect Characters
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                self.kernel_width,
                self.kernel_height,
            ),
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel,
        )

        # Find Contours
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        boxes = []

        for contour in contours:

            x, y, w, h = cv2.boundingRect(contour)

            if h < self.min_height:
                continue

            if w < self.min_width:
                continue

            boxes.append((x, y, w, h))

        # Sort Top -> Bottom
        boxes.sort(key=lambda b: b[1])

        lines = []

        for x, y, w, h in boxes:
            margin = 4

            x1 = max(0, x - margin)
            y1 = max(0, y - margin)

            x2 = min(image.shape[1], x + w + margin)
            y2 = min(image.shape[0], y + h + margin)

            crop = image[
                y1:y2,
                x1:x2,
            ]

            lines.append(
                {
                    "image": crop,
                    "bbox": (
                        x1,
                        y1,
                        x2,
                        y2,
                    ),
                }
            )

        return lines