from torch.utils.data import DataLoader
from dataset.collate import CTCCollate
from dataset.synth90k import Synth90kDataset
from dataset.vocabulary import Vocabulary
from torchvision import transforms
from preprocessing.transforms import train_transform
from utils.visualize import show_batch
from models.modules.cnn import CNNFeatureExtractor
from models.modules.sequence import SequenceConverter,BidirectionalLSTM
from models.modules.classifier import CTCClassifier
from models.recognizer import CRNN
from decoder.greedy import GreedyDecoder
from trainer.loss import CTCLossWrapper


vocab = Vocabulary(
    vocab_file="./resources/vocab.json",
)

train_dataset = Synth90kDataset(
    image_dir="data/images",
    annotation_file="data/train.txt",
    vocabulary=vocab,
    transform=train_transform,
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=CTCCollate(),
)

batch = next(iter(train_loader))

print(batch["images"].shape)
print(batch["targets"].shape)
print(batch["target_lengths"])

batch = next(iter(train_loader))

#show_batch(batch)

cnn = CNNFeatureExtractor()
x = batch["images"]
y = cnn(x)

print(y.shape)

converter = SequenceConverter()
features = cnn(batch["images"])
sequence = converter(features)
print(sequence.shape)


lstm = BidirectionalLSTM(
    input_size=512,
    hidden_size=256
)

output = lstm(sequence)

print(output.shape)

classifier = CTCClassifier(
    input_size=512,
    num_classes=vocab.num_classes
)
logits = classifier(output)
print(logits.shape)


model = CRNN(
    num_classes=vocab.num_classes
)

output = model(batch["images"])

print(output["logits"].shape)
print(output)

decoder = GreedyDecoder(vocab)
predictions = decoder.decode(
    output["logits"]
)
print(predictions[:5])

criterion = CTCLossWrapper()
loss = criterion(
    logits=output["logits"],
    targets=batch["targets"],
    input_lengths=output["input_lengths"],
    target_lengths=batch["target_lengths"]
)
print(loss)
loss.backward()
print("Backward ok")

for i in range(5):
    sample = train_dataset[i]
    print(sample["image"].shape)