import torch
import cv2
from pathlib import Path

from .evaluator import get_embedding
from .database import FaceDatabase
from .transforms import FaceTransform



class FaceRegister:
    def __init__(
            self,
            model,
            device
    ):

        self.model = model
        self.device = device
        self.transform = FaceTransform()

    def load_image(
            self,
            path
    ):
        image = cv2.imread(
            str(path)
        )
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )
        return self.transform(
            image
        )



    def create_embedding(
            self,
            folder
    ):
        embeddings = []
        images = Path(folder).glob("*")

        for image_path in images:
            image = self.load_image(
                image_path
            )
            embedding = get_embedding(
                self.model,
                image,
                self.device
            )
            embeddings.append(
                embedding
            )

        embeddings = torch.cat(
            embeddings,
            dim=0
        )
        mean_embedding = torch.mean(
            embeddings,
            dim=0,
            keepdim=True
        )
        mean_embedding = torch.nn.functional.normalize(
            mean_embedding,
            p=2,
            dim=1
        )

        return mean_embedding