import torch
import torch.nn.functional as F


def get_embedding(
        model,
        image,
        device
):
    model.eval()

    with torch.no_grad():
        if not torch.is_tensor(image):
            image = torch.tensor(
                image,
                dtype=torch.float32
            )

        image = image.unsqueeze(0)
        image = image.to(device)

        embedding = model(image)

        embedding = F.normalize(
            embedding,
            p=2,
            dim=1
        )
    return embedding.cpu()

def similarity(
        emb1,
        emb2
):
    return F.cosine_similarity(
        emb1,
        emb2
    ).item()