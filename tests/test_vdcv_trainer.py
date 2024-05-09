from unittest import TestCase


from app.options.train_options import TrainCompOptions
from app.trainers.vdcv_trainer import VDCVTrainer
from app.datasets.data_set import Dataset3class
from app.datasets.dataloader import build_dataloader


class TestVDCVTrainer(TestCase):

    def test_train(self):
        parser = TrainCompOptions()
        args = parser.parse()

        train_ds = Dataset3class(args.train_okay, args.train_bad, args.train_unknow)

        train_dataloader = build_dataloader(train_ds, 1, True)

        trainer = VDCVTrainer(args)

        # trainer.train(train_loader=train_dataloader, epochs=args.num_epochs)

        test_ds = Dataset3class(args.test_okay, args.test_bad, args.test_unknow)


        # test_dataloader = build_dataloader(test_ds, 1, False)
        # self.fail()
