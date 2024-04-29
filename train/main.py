import torch
import torch.nn as nn
import torch.nn.functional as F

from app.utils import image_formatter

import os

from tqdm import tqdm

from app.config.settings import Settings


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

        t_img = image_formatter.format_img(img_path)
        t_class_id = torch.tensor(class_id)

        return {'img': t_img, 'label': t_class_id}


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.act = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool2d(2,2)

        self.conv0 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.adaptivepool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64,10)
        self.linear2 = nn.Linear(10,3)

    def forward(self, X):
        out = self.conv0(X)
        out = self.act(out)
        out = self.maxpool(out)

        out = self.conv1(out)
        out = self.act(out)
        out = self.maxpool(out)

        out = self.conv2(out)
        out = self.act(out)
        out = self.maxpool(out)

        out = self.conv3(out)
        out = self.act(out)

        out = self.adaptivepool(out)
        out = self.flatten(out)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)

        return out


def accuracy(pred, label):
    answer = F.softmax(pred.detach()).numpy().argmax(1) == label.numpy().argmax(1)
    return answer.mean()


def train():
    global acc_current, loss_item
    train_ds = Dataset3class(Settings.train_okey_path, Settings.train_bad_path, Settings.train_unknow_path)
    test_ds = Dataset3class(Settings.test_okey_path, Settings.test_bad_path, Settings.test_unknow_path)

    batch_size = 1

    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True,
                                               batch_size=batch_size, num_workers=0,
                                               drop_last=True)

    test_loader = torch.utils.data.DataLoader(test_ds, shuffle=True,
                                              batch_size=batch_size, num_workers=0,
                                              drop_last=False)

    model = ConvNet()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9,
                                                                       0.999))

    for sample in train_loader:
        img = sample['img']
        label = sample['label']
        model(img)
        break

    epochs = 10

    for epoch in range(epochs):
        loss_val = 0
        acc_val = 0
        for sample in (pbar := tqdm(train_loader)):
            img, label = sample['img'], sample['label']

            optimizer.zero_grad()

            label = F.one_hot(label, 3).float()
            pred = model(img)

            loss = loss_fn(pred, label)

            loss.backward()
            loss_item = loss.item()
            loss_val = loss_item

            optimizer.step()

            acc_current = accuracy(pred, label)
            acc_val += acc_current

        pbar.set_description(f'loss: {loss_item: .5f}\taccuracy: {acc_current: .3f}')
        print(loss_val / len(train_loader))
        print(acc_val / len(train_loader))

    PATH = Settings.model_dir
    torch.save(model.state_dict(), PATH)


# train() todo refactoring

# def train():
