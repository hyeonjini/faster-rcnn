import albumentations as A
from albumentations.pytorch import ToTensorV2

class BaseAugmentations:
    def __init__(self, img_size: int, mean: float, std: float):

        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            ToTensorV2(),
            A.Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)
