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

    def __init__(self, split, crop_size, txt_root='./development_kit/data',
                 image_root='./images'):
        assert split in ('train', 'val', 'test'), "split must be one of: train, val, and test"
        self.image_root = image_root
        self.split = split
        # build transform
        self.train_transform = transforms.Compose([
            transforms.RandomCrop([crop_size, crop_size]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
        self.eval_transform = transforms.Compose([
            transforms.CenterCrop([crop_size, crop_size]),  # only center crop in eval
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
        self.training = split == 'train'
        # build image path list
        txt_dir = os.path.join(txt_root, split + '.txt')
        self.images = []
        self.labels = []
        with open(txt_dir, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                self.images.append(tokens[0])
                if len(tokens) > 1:
                    self.labels.append(int(tokens[1]))
        self.has_label = len(self.labels) > 0

    def __getitem__(self, idx):
        relative_path = self.images[idx]
        path = os.path.join(self.image_root, relative_path)
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        if self.training:
            tensor = self.train_transform(img)
        else:
            tensor = self.eval_transform(img)
        if self.has_label:
            return relative_path, tensor, self.labels[idx]
        else:
            return relative_path, tensor

    def __len__(self):
        return len(self.images)

    # methods to change train/eval mode
    def train(self, mode=True):
        self.training = mode

    def eval(self):
        self.train(False)
