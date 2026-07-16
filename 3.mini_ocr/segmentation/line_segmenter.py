from dataclasses import dataclass


@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int


class LineSegmenter:
    def segment(self, boxes):
        if len(boxes) == 0:
            return []

        boxes = sorted(boxes, key=lambda x: x.y)
        lines = []

        current_line = [boxes[0]]

        current_y = boxes[0].y + boxes[0].height / 2
        threshold = 15

        for box in boxes[1:]:
            box_center = box.y + box.height / 2
            if abs(box_center - current_y) <= threshold:
                current_line.append(box)
            else:
                lines.append(current_line)
                current_line = [box]
                current_y = box.y + box.height / 2

        lines.append(current_line)
        return lines
