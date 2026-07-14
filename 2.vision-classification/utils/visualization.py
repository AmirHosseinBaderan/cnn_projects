import matplotlib.pyplot as plt
import torch

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

def denormalize(img:torch.Tensor):
    img = img.clone()

    img = img * IMAGENET_STD[:,None,None]
    img = img * IMAGENET_MEAN[:,None,None]

    return img.clamp(0,1)

def show_image(image,label=None):

    image = denormalize(image)
    image = image.permute(1, 2, 0)

    plt.imshow(image)

    if label is not None:
        plt.title(label)

    plt.axis("off")
    plt.show()

def show_batch(images, labels, class_names):

    fig = plt.figure(figsize=(12, 8))

    for i in range(min(16, len(images))):
        plt.subplot(4, 4, i + 1)
        image = denormalize(images[i])
        image = image.permute(1, 2, 0)
        plt.imshow(image)
        plt.title(class_names[labels[i]])

        plt.axis("off")

    plt.tight_layout()
    plt.show()