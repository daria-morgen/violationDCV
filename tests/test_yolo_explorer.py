import os
from unittest import TestCase

from app.handlers.yolo_explorer import YoloPersonExplorer
from app.config.settings import Settings
import shutil

from app.utils.filepath_editor import get_predict_path


class TestYoloPersonExplorer(TestCase):

    def test_predict_and_save_create_user_dir(self):
        yolo_explorer = YoloPersonExplorer()
        user_id = 1
        img = Settings.parent_dir + '/train/darasets/test/okey/okey470.jpg'

        yolo_explorer.predict_and_save(img,
                                       Settings.project_predicts)

        self.assertTrue(os.path.isdir(Settings.project_predicts + "/predict"))

        # Delete a non-empty directory called 'thedirectory'
        shutil.rmtree(Settings.project_predicts + "/predict")

    def test_predict_and_save_predict_person(self):
        yolo_explorer = YoloPersonExplorer()
        user_id = 2
        img = Settings.parent_dir + '/train/darasets/test/okey/okey470.jpg'

        yolo_explorer.predict_and_save(
            img,
            Settings.project_predicts)

        self.assertTrue(len(Settings.project_predicts + '/predict/crops/person') > 0)

        # Delete a non-empty directory called 'thedirectory'
        shutil.rmtree(Settings.project_predicts + "/predict")

    def test_predict_and_save_user_person_not_found(self):
        yolo_explorer = YoloPersonExplorer()
        user_id = 3
        img = Settings.parent_dir + '/train/darasets/test/unknow/unknow97.jpg'

        with self.assertRaises(Exception):
            yolo_explorer.predict_and_save(img,
                                           Settings.project_predicts)

        shutil.rmtree(Settings.project_predicts + "/predict")

    # def test_predict_and_save_predict_person_on_video(self):
    #     yolo_explorer = YoloPersonExplorer()
    #     user_id = 73636
    #
    #     yolo_explorer.predict_and_save(
    #                                    'https://youtu.be/6yYchgX1fMw?si=OsEIsqdIPoyzST8R',
    #                                    get_predict_path_by_user(user_id))
    #
    #     result_path = os.path.join(Settings.project_predicts, str(user_id) + '/crops/person')
    #
    #     self.assertTrue(len(os.listdir(result_path)) == Settings.max_crop)
    #
    #     # Delete a non-empty directory called 'thedirectory'
    #     shutil.rmtree(os.path.join(Settings.project_predicts, str(user_id)))
