import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f'{self.img_labels.iloc[idx, 0]}.png')
        image = read_image(img_path)
        label_list = eval(self.img_labels.iloc[idx, 1])
        
        label_h_frac = label_list[0]
        label_v_frac = label_list[1]
        label_n_flow = label_list[2]

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'labels': {'label_h_frac': label_h_frac,
                                             'label_v_frac': label_v_frac,
                                             'label_n_flow': label_n_flow}}

        return sample


class SingleClassCustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f'{self.img_labels.iloc[idx, 0]}.png')
        image = read_image(img_path)
        label = int(self.img_labels.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'labels': label}

        return sample

