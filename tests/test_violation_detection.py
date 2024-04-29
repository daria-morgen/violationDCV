from unittest import TestCase
from app.handlers.violation_detection import ViolationDetection
from app.config.settings import Settings

import os



class TestViolationDetection(TestCase):

    def test_predict_okey(self):
        vp = ViolationDetection()

        img_path = os.path.join(Settings.parent_dir, 'train/darasets/test/okey/okey374.jpg')

        pred = vp.predict(img_path)

        self.assertEqual(pred, 0)

    def test_predict_bad(self):
        vp = ViolationDetection()

        img_path = os.path.join(Settings.parent_dir, 'train/darasets/test/bad/bad199.jpg')

        pred = vp.predict(img_path)

        self.assertEqual(pred, 1)
