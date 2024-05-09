from unittest import TestCase
from app.handlers.violation_detection_explorer import ViolationDetector
import os

from app.options.train_options import TrainCompOptions


class TestViolationDetection(TestCase):
    @classmethod
    def setUpClass(cls):
        parser = TrainCompOptions()
        cls.args = parser.parse()

    def test_predict_okay(self):
        vp = ViolationDetector(self.args)

        img = './data/okey374.jpg'

        pred = vp.predict(img)

        self.assertEqual(pred, 'okay')

    def test_predict_bad(self):
        vp = ViolationDetector(self.args)

        img = './data/bad199.jpg'

        pred = vp.predict(img)

        self.assertEqual(pred, 'bad')
