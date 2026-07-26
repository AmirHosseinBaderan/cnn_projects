from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ImageInfo:
    filename: str

    folder: str | None = None

    width: int | None = None

    height: int | None = None

    depth: int | None = None