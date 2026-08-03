from pathlib import Path
import xml.etree.ElementTree as ET

from domain import (
    Annotation,
    BoundingBox,
    ImageInfo,
    LabeledObject,
)

from dataset.parsers.base import AnnotationParser


class IRLPRXMLParser(AnnotationParser):

    def parse(
        self,
        annotation_path: str | Path,
    ) -> Annotation:

        root = ET.parse(Path(annotation_path)).getroot()

        annotation = Annotation(
            image=self._parse_image(root),
        )

        for object_element in root.findall("object"):
            annotation.add_object(
                self._parse_object(object_element),
            )

        return annotation

    def _parse_image(
        self,
        root: ET.Element,
    ) -> ImageInfo:

        size = root.find("size")

        return ImageInfo(
            filename=self._text(root, "filename"),
            folder=self._text(root, "folder"),
            width=self._optional_int(size, "width"),
            height=self._optional_int(size, "height"),
            depth=self._optional_int(size, "depth"),
        )

    def _parse_object(
        self,
        object_element: ET.Element,
    ) -> LabeledObject:

        bbox = object_element.find("bndbox")

        if bbox is None:
            raise ValueError("Missing <bndbox> element.")

        return LabeledObject(
            label=self._text(object_element, "name"),
            bbox=BoundingBox(
                xmin=self._number(bbox, "xmin"),
                ymin=self._number(bbox, "ymin"),
                xmax=self._number(bbox, "xmax"),
                ymax=self._number(bbox, "ymax"),
            ),
        )

    @staticmethod
    def _text(
        element: ET.Element,
        tag: str,
    ) -> str | None:

        value = element.findtext(tag)

        if value is None:
            return None

        return value.strip()

    @staticmethod
    def _number(
        element,
        tag,
    ) -> float:
        value = element.findtext(tag)

        if value is None:
            raise ValueError(f"{tag} not found")

        return float(value)

    @staticmethod
    def _optional_int(
        element: ET.Element | None,
        tag: str,
    ) -> int | None:

        if element is None:
            return None

        value = element.findtext(tag)

        if value is None:
            return None

        return int(value)