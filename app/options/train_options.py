from .base_options import BaseOptions
import argparse


class TrainCompOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
        self.parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU')

        self.is_train = True