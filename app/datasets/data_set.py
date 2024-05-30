import os

import cv2
import torch

import numpy as np


def format_img(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img / 255.0

    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    img = img.transpose((2, 0, 1))

    t_img = torch.from_numpy(img)

    return t_img


class Dataset3class(torch.utils.data.Dataset):
    def __init__(self, path_dir1: str, path_dir2: str, path_dir3: str):
        super().__init__()

        self.path_dir1 = path_dir1
        self.path_dir2 = path_dir2
        self.path_dir3 = path_dir3

        self.dir1_list = sorted(os.listdir(path_dir1))
        self.dir2_list = sorted(os.listdir(path_dir2))
        self.dir3_list = sorted(os.listdir(path_dir3))

    def __len__(self):
        return len(self.dir1_list) + len(self.dir2_list) + len(self.dir3_list)

    def __getitem__(self, idx):
        if idx < len(self.dir1_list):
            class_id = 0
            img_path = os.path.join(self.path_dir1, self.dir1_list[idx])
        elif len(self.dir1_list) <= idx < (len(self.dir1_list) + len(self.dir2_list)):
            class_id = 1
            idx -= len(self.dir1_list)
            img_path = os.path.join(self.path_dir2, self.dir2_list[idx])
        else:
            class_id = 2
            idx -= len(self.dir1_list) + len(self.dir2_list)
            img_path = os.path.join(self.path_dir3, self.dir3_list[idx])

        t_img = format_img(img_path)
        t_class_id = torch.tensor(class_id)

        return {'img': t_img, 'label': t_class_id}
