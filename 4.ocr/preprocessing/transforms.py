from torchvision import transforms

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 128

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5,),
        std=(0.5,)
    )
])

valid_transform = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5,),
        std=(0.5,)
    )
])
