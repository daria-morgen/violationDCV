import torch
import torch.nn as nn
import torch.nn.functional as F

from app.handlers.models.violation_detection import ConvNet

from tqdm import tqdm


class VDCVTrainer(object):
    def __init__(self, args):
        self.opt = args
        self.model = ConvNet()

    def train(self, train_loader, epochs):
        global acc_current, loss_item

        model = self.model

        loss_fn = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.opt.lr, betas=(0.9,
                                                                           0.999))

        for sample in train_loader:
            img = sample['img']
            label = sample['label']
            model(img)
            break

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

                acc_current = self.accuracy(pred, label)
                acc_val += acc_current

            pbar.set_description(f'loss: {loss_item: .5f}\taccuracy: {acc_current: .3f}')
            print(loss_val / len(train_loader))
            print(acc_val / len(train_loader))

        PATH = self.opt.model_dir
        torch.save(model.state_dict(), PATH)

    @staticmethod
    def accuracy(pred, label):
        answer = F.softmax(pred.detach()).numpy().argmax(1) == label.numpy().argmax(1)
        return answer.mean()