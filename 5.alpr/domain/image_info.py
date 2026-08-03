from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ImageInfo:
    filename: str

    folder: str | None = None

    width: float | None = None

    height: float | None = None

    depth: float | None = None