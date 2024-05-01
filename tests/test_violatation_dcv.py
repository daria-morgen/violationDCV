import shutil
from unittest import TestCase

from app.handlers.violatation_dcv import ViolatationDCV
from app.config.settings import Settings


class TestViolatationDCV(TestCase):

    def test_detect_fail_no_persons(self):
        app = ViolatationDCV()

        with self.assertRaises(Exception):
            app.detect(Settings.parent_dir + '/train/darasets/test/unknow/unknow88.jpg', Settings.project_predicts)

    def test_detect(self):
        app = ViolatationDCV()

        result = app.detect(Settings.parent_dir + '/train/darasets/test/okey/okey357.jpg', Settings.project_predicts)

        self.assertEqual(result[0].get('label'), 'okay')
