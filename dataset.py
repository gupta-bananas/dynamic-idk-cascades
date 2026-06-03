# Handing the downloading and loading ImageNet Datasets

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_imagenet_v2_loaders(data_dir="./data", batch_size=32):
    """ Setting up data pipeline for ImageNet-V2"""

    """Standered normalization transformation for ImageNet-trained models"""

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
    ])

    if os.path.exists(data_dir) or os.path.exists(os.path.join(data_dir, "test")):
        print(f"Loading ImageNet-V2 data from: {data_dir}")
        test_dataset = datasets.ImageFolder(
            root=os.path.join(data_dir, "test"),
            transform=transforms
        )


    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False

    )

    return test_loader