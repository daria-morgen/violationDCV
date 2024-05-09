import os
import shutil
from unittest import TestCase

from app.handlers.violation_dcv import ViolationDCV
from app.options.train_options import TrainCompOptions


class TestViolatationDCV(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project = './data/project_predicts'
        if not os.path.isdir(cls.project):
            os.mkdir(cls.project)

        parser = TrainCompOptions()
        cls.args = parser.parse()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.project)

    def test_detect_fail_no_persons(self):
        app = ViolationDCV(self.args)
        project = self.project

        if not os.path.isdir(project):
            os.mkdir(project)

        with self.assertRaises(Exception):
            app.detect('./data/unknow88.jpg', project)

    def test_detect(self):
        app = ViolationDCV(self.args)
        project = self.project

        result = app.detect('./data/okey470.jpg', project)

        self.assertEqual(result[0].get('label'), 'okay')

