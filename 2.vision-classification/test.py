from datasets.test_dataset import test_dataset
import torch
from models.simple_cnn import SimpleCNN

test_dataset()

x = torch.randn(4, 3, 150, 150)

model = SimpleCNN()

output = model(x)

print(output)