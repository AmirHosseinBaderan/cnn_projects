from dataclasses import dataclass

from domain.bounding_box import BoundingBox


@dataclass(slots=True, frozen=True)
class LabeledObject:
    label: str

    bbox: BoundingBox