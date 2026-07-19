import json


class Vocabulary:
    def __init__(self, vocab_file):
        with open(vocab_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.blank_idx = data["blank"]

        self.characters = data["characters"]

        self.char_to_idx = {
            c: i + 1
            for i, c in enumerate(self.characters)
        }

        self.idx_to_char = {
            i + 1: c
            for i, c in enumerate(self.characters)
        }

    @property
    def num_classes(self):
        return len(self.characters) + 1

    def encode(self, text):
        return [
            self.char_to_idx[c]
            for c in text
        ]

    def decode(self, indices):
        result = []
        previous = None

        for idx in indices:
            if idx == self.blank_idx:
                previous = idx
                continue

            if idx == previous:
                continue

            result.append(
                self.idx_to_char[idx]
            )

            previous = idx

        return "".join(result)