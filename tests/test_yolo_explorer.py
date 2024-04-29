import os
from unittest import TestCase

from app.handlers.yolo_explorer import YoloHumanExplorer
from app.config.settings import Settings
import shutil


class TestYoloHumanExplorer(TestCase):

    def test_predict_and_save_create_user_dir(self):
        yolo_explorer = YoloHumanExplorer()
        user_id = 73636264

        yolo_explorer.predict_and_save(user_id,
                                       'https://img.freepik.com/free-photo/confident-smiling-man-training-in-gym-flex-strong-biceps-show-muscles_176420-17997.jpg')

        result_path = os.path.join(Settings.parent_dir, str(user_id))

        self.assertTrue(os.path.isdir(result_path))

        # Delete a non-empty directory called 'thedirectory'
        shutil.rmtree(result_path)

    def test_predict_and_save_predict_human(self):
        yolo_explorer = YoloHumanExplorer()
        user_id = 73636264

        yolo_explorer.predict_and_save(user_id,
                                       'https://img.freepik.com/free-photo/confident-smiling-man-training-in-gym-flex-strong-biceps-show-muscles_176420-17997.jpg')

        result_path = os.path.join(Settings.parent_dir, str(user_id) + '/predict/crops/person')

        self.assertTrue(len(os.listdir(result_path)) > 0)

        # Delete a non-empty directory called 'thedirectory'
        shutil.rmtree(os.path.join(Settings.parent_dir, str(user_id)))


    def test_predict_and_save_predict_human_on_video(self):
        yolo_explorer = YoloHumanExplorer()
        user_id = 73636

        yolo_explorer.predict_and_save(user_id,
                                       'https://youtu.be/LNwODJXcvt4?si=cypiuxu9sijT9ZUE')

        result_path = os.path.join(Settings.parent_dir, str(user_id) + '/predict/crops/person')

        self.assertTrue(len(os.listdir(result_path)) == Settings.max_crop)

        # Delete a non-empty directory called 'thedirectory'
        shutil.rmtree(os.path.join(Settings.parent_dir, str(user_id)))