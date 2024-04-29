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
