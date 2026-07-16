class WordSegmenter:
    def segment(self, line):
        if not line:
            return []

        line = sorted(line, key=lambda b: b.x)
        words = []

        current_word = [line[0]]
        previous_right = line[0].x + line[0].width

        threshold = 20
        for box in line[1:]:
            gap = box.x - previous_right
            if gap <= threshold:
                current_word.append(box)
            else:
                words.append(current_word)
                current_word = [box]

            previous_right = box.x + box.width

        words.append(current_word)
        return words
