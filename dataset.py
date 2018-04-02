import os
import glob
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image


class MiniplacesDataset(data.Dataset):
    num_classes = 100
    mean = [0.45432566120701207, 0.4362440955619064, 0.40466656402468215]
    std = [0.2643629213905866, 0.2612573977850761, 0.2790479352583138]

    def __init__(self, split, crop_size, label_root='./labels',
                 image_root='./images'):
        assert split in ('train', 'val', 'test'), "split must be one of: train, val, and test"
        image_dir = os.path.join(image_root, split)
        self.split = split
        # build transform
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomCrop([crop_size, crop_size]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.RandomCrop([crop_size, crop_size]),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])
        # build image path list
        if split == 'test':
            self.images = glob.glob(os.path.join(image_dir, '*'))
        else:
            label_dir = os.path.join(label_root, split + '.txt')
            with open(label_dir, 'r') as f:
                self.images = []
                self.labels = []
                for line in f:
                    image, label = line.strip().split()
                    label = int(label)
                    self.images.append(os.path.join(image_root, image))
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        tensor = self.transform(img)# * 255 - torch.Tensor([123, 117, 104]).view(3, 1, 1)
        if self.split == 'test':
            return tensor
        else:
            return tensor, self.labels[idx]
