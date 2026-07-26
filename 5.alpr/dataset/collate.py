import torch


class DetectorCollate:

    def __call__(self, batch):

        images = torch.stack(
            [sample["image"] for sample in batch]
        )

        annotations = [
            sample["annotation"]
            for sample in batch
        ]

        return {
            "images": images,
            "annotations": annotations,
        }