import torch
import torch.nn.functional as F


class FaceSearch:

    def __init__(
            self,
            database
    ):
        self.database = database

    def search(
            self,
            query_embedding,
            threshold=0.7
    ):

        best_person = None
        best_score = -1.0

        query_embedding = query_embedding.cpu()

        for person, embeddings in self.database.items():

            for embedding in embeddings:

                embedding = embedding.cpu()

                score = F.cosine_similarity(
                    query_embedding,
                    embedding,
                    dim=0
                ).item()

                if score > best_score:

                    best_score = score
                    best_person = person

        if best_score < threshold:

            return {
                "name": "Unknown",
                "score": best_score
            }

        return {
            "name": best_person,
            "score": best_score
        }