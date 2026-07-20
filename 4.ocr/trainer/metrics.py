from decoder.greedy import GreedyDecoder


class OCRMetrics:

    def __init__(self, vocabulary):
        self.decoder = GreedyDecoder(vocabulary)

    def evaluate_batch(self, logits, labels, max_samples=5):

        predictions = self.decoder.decode(logits)

        results = []

        for gt, pred in zip(labels[:max_samples], predictions[:max_samples]):
            results.append({
                "gt": gt,
                "pred": pred
            })

        return results