from .evaluator import get_embedding


class FaceRecognizer:

    def __init__(
            self,
            detector,
            model,
            search_engine,
            device
    ):

        self.detector = detector
        self.model = model
        self.search_engine = search_engine
        self.device = device

    def recognize(
            self,
            image
    ):

        detections = self.detector.detect(
            image
        )

        results = []

        for detection in detections:

            embedding = get_embedding(
                model=self.model,
                image=detection["face"],
                device=self.device
            )

            search_result = self.search_engine.search(
                embedding
            )

            results.append(
                {
                    "name": search_result["name"],
                    "score": search_result["score"],
                    "box": detection["box"],
                    "confidence": detection["confidence"],
                    "center": detection["center"],
                    "size": detection["size"]
                }
            )

        return results