from dataclasses import dataclass, field

from domain.image_info import ImageInfo
from domain.labeled_object import LabeledObject


@dataclass(slots=True)
class Annotation:
    image: ImageInfo
    objects: list[LabeledObject] = field(default_factory=list)

    def add_object(
        self,
        obj: LabeledObject,
    ) -> None:
        self.objects.append(obj)

    def has_label(
        self,
        label: str,
    ) -> bool:
        return any(
            obj.label == label
            for obj in self.objects
        )

    def get_by_label(
        self,
        label: str,
    ) -> list[LabeledObject]:
        return [
            obj
            for obj in self.objects
            if obj.label == label
        ]

    def exclude_label(
        self,
        label: str,
    ) -> list[LabeledObject]:
        return [
            obj
            for obj in self.objects
            if obj.label != label
        ]

    def first(
        self,
        label: str,
    ) -> LabeledObject | None:

        for obj in self.objects:
            if obj.label == label:
                return obj

        return None