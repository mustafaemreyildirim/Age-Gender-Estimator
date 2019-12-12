import torch
import random
import glob
import os
import sys
from PIL import Image

class IMDBWIKI(torch.utils.data.Dataset):

    def __init__(self, root, train=True, transform=None, seed=0):
        random.seed(seed)

        self.root = root
        self.train = train
        self.transform = transform

        self.images = []
        for i, path in enumerate(glob.glob(os.path.abspath('{}/**/*.jpg'.format(self.root)))):
            age, gender = os.path.basename(path).split('_')[:2]
            age, gender = int(age), int(gender)
            if age > 0 and age < 100 and (gender == 0 or gender == 1):
                self.images.append(path)

        random.shuffle(self.images)
        if train:
            self.images = self.images[:-2000]
        else:
            self.images = self.images[-2000:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        age, gender = os.path.basename(path).split('_')[:2]
        age, gender = float(age) / 100.0, int(gender)

        image = Image.open(path)
        if self.transform:
            image = self.transform(image)

        return image, (gender, age)

