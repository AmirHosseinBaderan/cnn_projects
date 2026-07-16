import torch
import torch.nn.functional as F


class FaceSearch:
    def __init__(self, embeddings, names):
        self.embeddings = embeddings
        self.names = names

    def search(
            self,
            query_embedding,
            threshold=0.7
    ):
        similarities = F.cosine_similarity(
            query_embedding,
            self.embeddings
        )

        best_score, index = torch.max(
            similarities,
            dim=0
        )

        score = best_score.item()

        if score < threshold:
            return {
                "name": "Unknown",
                "score": score
            }

        return {
            "name": self.names[str(index.item())],
            "score": score
        }
