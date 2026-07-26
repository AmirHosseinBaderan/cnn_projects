from dataset.detector_dataset import DetectorDataset
from dataset.parsers import IRLPRXMLParser


dataset = DetectorDataset(
    root="data/car_images/train",
    parser=IRLPRXMLParser(),
)

print(len(dataset))

sample = dataset[0]
print(sample)

plate = sample["annotation"].first("کل ناحیه پلاک")
print(plate)