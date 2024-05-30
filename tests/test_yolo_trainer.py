from unittest import TestCase

from app.trainers.yolo_trainer import YOLOTrainer


class TestYOLOTrainer(TestCase):
    def test_train(self):

        trainer = YOLOTrainer()

        trainer.train()


        self.fail()
