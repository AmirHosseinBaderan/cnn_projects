import torch
import torch.nn.functional as F


def cosine_similarity(
        embedding1,
        embedding2
):
    return F.cosine_similarity(
        embedding1,
        embedding2,
    ).item()
