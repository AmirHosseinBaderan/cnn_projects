import cv2
import numpy as np
from ..preprocessing.grayscale import to_grayscale
from ..preprocessing.threshold import otsu_threshold
from ..detection.connected_components import ConnectedComponentDetector


class CharacterSegmenter:

    def __init__(self):
        self.connected_detector = ConnectedComponentDetector()
        self.min_character_area = 10

    def segment(self, roi: np.ndarray):
        if len(roi.shape) == 3:
            roi = to_grayscale(roi)

        binary = otsu_threshold(roi)

        num_labels, _, stats, _ = self.connected_detector.detect(binary)

        characters = []
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]

            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            area = stats[i, cv2.CC_STAT_AREA]

            if area < self.min_character_area:
                continue

            character = roi[
                y:y + h,
                x:x + w
            ]
            characters.append({
                "image": character,
                "x": x,
                "y": y,
                "width": w,
                "height": h,
            })

        characters.sort(key=lambda c: c["x"])
        return characters
