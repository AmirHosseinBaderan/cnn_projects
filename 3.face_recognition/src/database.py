import os
import json
import torch


class FaceDatabase:

    def __init__(
            self,
            path="./database",
            embedding_dim=128
    ):
        self.path = path
        self.embedding_dim = embedding_dim
        os.makedirs(
            self.path,
            exist_ok=True
        )
        self.embedding_path = os.path.join(
            self.path,
            "embeddings.pt"
        )
        self.names_path = os.path.join(
            self.path,
            "persons.json"
        )
        self.initialize()

    def initialize(self):
        # Create embeddings file
        if not os.path.exists(
                self.embedding_path
        ):
            torch.save(
                torch.empty(
                    (0, self.embedding_dim)
                ),
                self.embedding_path
            )
        # Create names file
        if not os.path.exists(
                self.names_path
        ):
            with open(
                    self.names_path,
                    "w"
            ) as f:
                json.dump(
                    {},
                    f,
                    indent=4
                )

    def load(self):
        embeddings = torch.load(
            self.embedding_path
        )
        with open(
                self.names_path,
                "r"
        ) as f:
            names = json.load(f)
        return embeddings, names

    def save(
            self,
            embeddings,
            names
    ):
        torch.save(
            embeddings,
            self.embedding_path
        )

        with open(
                self.names_path,
                "w"
        ) as f:
            json.dump(
                names,
                f,
                indent=4
            )

    def add_person(
            self,
            name,
            embedding
    ):
        embeddings, names = self.load()
        # Check duplicate
        if name in names.values():
            raise Exception(
                f"Person {name} already exists"
            )

        index = len(names)
        names[str(index)] = name
        embeddings = torch.cat(
            [
                embeddings,
                embedding
            ],
            dim=0
        )
        self.save(
            embeddings,
            names
        )
        return index