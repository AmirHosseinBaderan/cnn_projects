import torch
import torch.nn.functional as F


class CTCCollate:

    def __call__(self, batch):

        images = []
        targets = []
        target_lengths = []
        image_widths = []
        labels = []

        #
        # پیدا کردن بیشترین عرض داخل Batch
        #
        max_width = max(
            sample["image"].shape[-1]
            for sample in batch
        )

        for sample in batch:

            image = sample["image"]

            #
            # ذخیره عرض واقعی
            #
            image_widths.append(
                image.shape[-1]
            )

            #
            # Padding سمت راست
            #
            pad = max_width - image.shape[-1]

            image = F.pad(
                image,
                (0, pad),
                value=0.0,
            )

            images.append(image)

            target = torch.tensor(
                sample["target"],
                dtype=torch.long,
            )

            targets.append(target)

            target_lengths.append(
                len(target)
            )

            labels.append(
                sample["label"]
            )

        images = torch.stack(images)

        targets = torch.cat(targets)

        target_lengths = torch.tensor(
            target_lengths,
            dtype=torch.long,
        )

        image_widths = torch.tensor(
            image_widths,
            dtype=torch.long,
        )

        return {
            "images": images,
            "targets": targets,
            "labels": labels,
            "target_lengths": target_lengths,
            "image_widths": image_widths,
        }