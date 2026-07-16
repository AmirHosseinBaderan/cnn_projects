import torch

from src.similarity import cosine_similarity


a = torch.tensor(
    [[0.5,0.5,0.5]]
)


b = torch.tensor(
    [[0.51,0.49,0.5]]
)


score = cosine_similarity(
    a,
    b
)


print(score)