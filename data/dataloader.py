import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

class TinyImageNetDataLoader:
    def __init__(self, data_dir='dataset/tiny_imagenet/tiny-imagenet-200', batch_size=64):
        self.data_dir = data_dir
        self.batch_size = batch_size

        # Definizione delle trasformazioni
        self.transform = T.Compose([
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(0.5),
            T.Resize((224, 224)),  # Resize 
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_dataloaders(self):
        #return the dataloaders for training and validation
        train_dataset = ImageFolder(root=f'{self.data_dir}/train', transform=self.transform)
        val_dataset = ImageFolder(root=f'{self.data_dir}/val', transform=self.transform)
        print(f"Length of train dataset: {len(train_dataset)}")
        print(f"Length of val dataset: {len(val_dataset)}")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader
