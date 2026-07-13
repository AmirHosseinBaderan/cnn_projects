from models.cnn import build_cnn

from models.resnet import build_resnet

from models.efficientnet import (
    build_efficientnet
)


class ModelFactory:

    @staticmethod
    def create(
        model_name,
        num_classes=10,
        pretrained=True

    ):

        model_name = model_name.lower()

        if model_name == "cnn":
            return build_cnn()

        if model_name == "resnet18":
            return build_resnet(
                num_classes,
                pretrained
            )

        if model_name == "efficientnet_b0":
            return build_efficientnet(
                num_classes,
                pretrained
            )

        raise ValueError(
            f"Unknown Model : {model_name}"
        )