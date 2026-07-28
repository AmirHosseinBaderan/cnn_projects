import os
import cv2

print(f"[DEBUG] Visualizer module loaded. DISPLAY={os.environ.get('DISPLAY')}")
os.environ['QT_QPA_PLATFORM'] = 'xcb'


class Visualizer:

    def draw(
        self,
        image,
        detections,
    ):
        print(f"[DEBUG] Visualizer.draw called with {len(detections)} detections")
        print(f"[DEBUG] DISPLAY={os.environ.get('DISPLAY')}")

        image = image.copy()

        for detection in detections:

            bbox = detection["bbox"]

            cv2.rectangle(
                image,
                (
                    int(bbox.xmin),
                    int(bbox.ymin),
                ),
                (
                    int(bbox.xmax),
                    int(bbox.ymax),
                ),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                image,
                f"{detection['score']:.2f}",
                (
                    int(bbox.xmin),
                    int(bbox.ymin - 5),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        return image

    def draw(
        self,
        image,
        detections,
    ):

        image = image.copy()

        for detection in detections:

            bbox = detection["bbox"]

            cv2.rectangle(
                image,
                (
                    int(bbox.xmin),
                    int(bbox.ymin),
                ),
                (
                    int(bbox.xmax),
                    int(bbox.ymax),
                ),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                image,
                f"{detection['score']:.2f}",
                (
                    int(bbox.xmin),
                    int(bbox.ymin - 5),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        return image