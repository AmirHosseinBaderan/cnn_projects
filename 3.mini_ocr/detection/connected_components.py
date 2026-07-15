import numpy as np
import cv2


class ConnectedComponentDetector:

    def detect(self, image: np.ndarray):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            image,
            connectivity=8,
        )

        return num_labels, labels, stats, centroids
