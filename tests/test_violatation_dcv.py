import shutil
from unittest import TestCase

from app.handlers.violatation_dcv import ViolatationDCV
from app.config.settings import Settings

from app.utils.filepath_editor import get_predict_path_by_user


class TestViolatationDCV(TestCase):

    def test_detect_fail_no_persons(self):
        app = ViolatationDCV()
        save_path = get_predict_path_by_user(1)

        with self.assertRaises(Exception):
            app.detect(Settings.parent_dir + '/train/darasets/test/unknow/unknow88.jpg', save_path)

    def test_detect(self):
        app = ViolatationDCV()
        user_id = 2
        save_path = get_predict_path_by_user(user_id)

        result = app.detect(Settings.parent_dir + '/train/darasets/test/okey/okey357.jpg', save_path)

        self.assertEqual(result[0].get('label'), 'okay')

        shutil.rmtree(save_path)
