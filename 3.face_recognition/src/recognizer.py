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
            image_path
    ):

        detections = self.detector.detect(
            image_path
        )

        results = []

        for detection in detections:

            face = detection["face"]

            embedding = get_embedding(
                self.model,
                face,
                self.device
            )

            result = self.search_engine.search(
                embedding
            )

            results.append(
                {
                    "name": result["name"],
                    "score": result["score"],
                    "box": detection["box"],
                    "confidence": detection["confidence"]
                }
            )

        return results