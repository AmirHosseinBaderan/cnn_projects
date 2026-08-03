from abc import ABC, abstractmethod
from pathlib import Path

from domain.annotation import Annotation


class AnnotationParser(ABC):

    @abstractmethod
    def parse(
        self,
        annotation_path: str |Path,
    ) -> Annotation:
        raise NotImplementedError