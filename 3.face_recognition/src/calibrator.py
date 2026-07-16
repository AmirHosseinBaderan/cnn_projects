import numpy as np


class ThresholdCalibrator:
    def __init__(
            self,
            same_scores,
            different_scores,
    ):
        self.same_scores = same_scores
        self.different_scores = different_scores

    def find_best_threshold(self):
        best_threshold = 0
        best_accuracy = 0

        thresholds = np.arange(
            -1,
            1.001,
            0.001
        )

        for threshold in thresholds:
            tp = np.sum(
                self.same_scores >= threshold
            )

            tn = np.sum(
                self.different_scores < threshold
            )

            accuracy = (
                               tp + tn
                       ) / (
                               len(self.same_scores)
                               +
                               len(self.different_scores)
                       )

            if accuracy > best_accuracy:
                best_threshold = threshold
                best_accuracy = accuracy

        return best_threshold, best_accuracy
