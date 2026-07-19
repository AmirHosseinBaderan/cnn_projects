import torch


class CTCCollate:

    def __call__(self, batch):
        images = []
        targets = []
        target_lengths = []
        labels = [
            sample["label"]
            for sample in batch
        ]

        for sample in batch:
            images.append(sample["image"])

            target = torch.tensor(
                sample["target"],
                dtype=torch.long
            )

            targets.append(target)

            target_lengths.append(
                len(target)
            )

        images = torch.stack(images)

        targets = torch.cat(targets)

        target_lengths = torch.tensor(
            target_lengths,
            dtype=torch.long
        )

        return {
            "images": images,
            "targets": targets,
            "labels": labels,
            "target_lengths": target_lengths
        }
