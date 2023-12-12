import numpy as np

from PIL import Image

import albumentations as A

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import config

class ImgRecogDataset:
    def __init__(self, images, labels, resize, split, max_len) -> None:
        self.images = images
        self.labels = labels
        self.resize = resize
        self.split = split
        self.max_len = max_len
        self.transfroms = A.Compose([
            A.ToGray(always_apply=True),
            A.Normalize(always_apply=True)
        ])
        
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        img = Image.open(f"{config.DATA_PATH}/{self.split}/txt_images/{image}")
        if self.resize is not None:
            img = img.resize(
                (self.resize[1], self.resize[0]), resample=Image.BILINEAR
            )

        img_arr = np.array(img)
        
        transformed = self.transfroms(image=img_arr)
        
        img = transformed['image']
        #channel first
        img = np.transpose(img, (2, 0, 1))
        
        return {
            "image": torch.tensor(img, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long)
        }
        
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
        
    def __call__(self, batch):        
        labels = [item['label'] for item in batch]
        
        label_len = torch.tensor([len(item['label']) for item in batch])
        
        # pad all the labels to max length of the batch
        padded_labels = pad_sequence(labels, batch_first=True,
                                    padding_value=self.pad_idx)
        
        images = torch.tensor(np.array([item['image'].numpy() for item in batch]))
        
        return {'image': images, 'label': padded_labels, 'label_len':label_len}
        
def get_loaders(image_paths, labels, resize, max_len, split, batch_size, shuffle=False):
    data = ImgRecogDataset(image_paths, labels, resize, split, max_len)
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=MyCollate(config.PAD_VAL))
    return loader