import random
from pathlib import Path



class EvaluationDataset:
    def __init__(self, root):

        self.root = Path(root)
        self.people = {}
        for person in self.root.iterdir():
            if person.is_dir():
                images = list(
                    person.glob("*")
                )
                if len(images) >= 2:

                    self.people[
                        person.name
                    ] = images

        self.names = list(
            self.people.keys()
        )

    def create_pair(self):
        same = random.choice(
            [True, False]
        )

        if same:
            person = random.choice(
                self.names
            )
            img1, img2 = random.sample(
                self.people[person],
                2
            )
            return img1, img2, 1
        else:
            person1, person2 = random.sample(
                self.names,
                2
            )
            img1 = random.choice(
                self.people[person1]
            )
            img2 = random.choice(
                self.people[person2]
            )
            return img1, img2, 0