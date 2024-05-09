from .data_set import Dataset3class
import torch


def build_dataloader(dataset: Dataset3class, batch_size, drop_last):
    loader = torch.utils.data.DataLoader(dataset, shuffle=True,
                                         batch_size=batch_size, num_workers=0,
                                         drop_last=drop_last)

    return loader
