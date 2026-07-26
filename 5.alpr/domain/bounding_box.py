from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def width(self) -> int:
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        return self.ymax - self.ymin

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.xmin + self.xmax) / 2,
            (self.ymin + self.ymax) / 2,
        )

    def to_xyxy(self) -> tuple[int, int, int, int]:
        return (
            self.xmin,
            self.ymin,
            self.xmax,
            self.ymax,
        )

    def to_xywh(self) -> tuple[int, int, int, int]:
        return (
            self.xmin,
            self.ymin,
            self.width,
            self.height,
        )