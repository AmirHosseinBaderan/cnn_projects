from .transforms import Transform
import torch

class ToTensor(Transform):
    def __call__(self, image, annotation):
        image = torch.from_numpy(image)
        image = image.permute(
            2,0,1
        ).float()
        
        return image,annotation