"""
Image transforms for training and evaluation.
"""

from typing import Dict, Any, Optional

from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(
    img_size: int = 224,
    mean: tuple = IMAGENET_MEAN,
    std: tuple = IMAGENET_STD,
    augmentation: bool = True,
) -> transforms.Compose:
    """
    Get training transforms with augmentation.
    """
    transform_list = [
        transforms.Resize((img_size, img_size)),
    ]
    
    if augmentation:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
            ),
        ])
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    return transforms.Compose(transform_list)


def get_eval_transforms(
    img_size: int = 224,
    mean: tuple = IMAGENET_MEAN,
    std: tuple = IMAGENET_STD,
) -> transforms.Compose:
    """
    Get evaluation transforms (no augmentation).
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Denormalize a tensor image for visualization.
    """
    import torch
    
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    return tensor * std + mean
