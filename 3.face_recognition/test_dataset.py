import torch

from src.database import FaceDatabase



db = FaceDatabase()



embedding = torch.rand(
    1,
    128
)



db.add_face(
    "Amir",
    embedding
)



db.add_face(
    "Amir",
    torch.rand(
        1,
        128
    )
)



db.add_face(
    "Ali",
    torch.rand(
        1,
        128
    )
)



data = db.load()


for person, embeddings in data.items():

    print(
        person,
        len(embeddings)
    )