import os
import torch


class FaceDatabase:

    def __init__(
            self,
            path="./data/database"
    ):

        self.path = path

        os.makedirs(
            self.path,
            exist_ok=True
        )

        self.file = os.path.join(
            self.path,
            "faces.pt"
        )

        self.initialize()

    def initialize(self):

        if not os.path.exists(
                self.file
        ):
            torch.save(
                {},
                self.file
            )

    def load(self):

        return torch.load(
            self.file
        )

    def save(
            self,
            data
    ):

        torch.save(
            data,
            self.file
        )

    def add_face(
            self,
            person_name,
            embedding
    ):

        database = self.load()

        if person_name not in database:
            database[person_name] = []

        database[person_name].append(
            embedding.cpu()
        )

        self.save(
            database
        )

    def get_persons(self):

        database = self.load()

        return list(
            database.keys()
        )
