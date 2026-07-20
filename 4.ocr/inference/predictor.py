import torch

class Predictor:
    def __init__(
        self,
        model,
        decoder,
        device,
    ):
        self.model = model
        self.decoder = decoder
        self.device = device

    @torch.no_grad()
    def predict(
        self,
        image,
    ):
        image = image.to(self.device)
        output = self.model(image)
        text = self.decoder.decode(
            output["logits"]
        )

        return text[0]