import os
from unittest import TestCase

from app.handlers.yolo_explorer import YoloPersonExplorer
from app.config.settings import Settings
import shutil

from app.utils.filepath_editor import get_predict_path_by_user, get_crop_path_by_user


class TestYoloPersonExplorer(TestCase):

    def test_predict_and_save_create_user_dir(self):
        yolo_explorer = YoloPersonExplorer()
        user_id = 1
        img = Settings.parent_dir + '/train/darasets/test/okey/okey470.jpg'

        yolo_explorer.predict_and_save(img,
                                       get_predict_path_by_user(user_id))

        result_path = get_predict_path_by_user(user_id)

        self.assertTrue(os.path.isdir(result_path))

        # Delete a non-empty directory called 'thedirectory'
        shutil.rmtree(result_path)

    def test_predict_and_save_predict_person(self):

        yolo_explorer = YoloPersonExplorer()
        user_id = 2
        img = Settings.parent_dir + '/train/darasets/test/okey/okey470.jpg'

        yolo_explorer.predict_and_save(
            img,
            get_predict_path_by_user(user_id))

        crop_path = get_crop_path_by_user(user_id)

        self.assertTrue(len(os.listdir(crop_path)) > 0)

        # Delete a non-empty directory called 'thedirectory'
        # shutil.rmtree(os.path.join(Settings.predict_dir, str(user_id)))

    def test_predict_and_save_user_person_not_found(self):

        yolo_explorer = YoloPersonExplorer()
        user_id = 3
        img = Settings.parent_dir + '/train/darasets/test/unknow/unknow97.jpg'

        with self.assertRaises(Exception):
            yolo_explorer.predict_and_save(img,
                                           get_predict_path_by_user(user_id))

    # def test_predict_and_save_predict_person_on_video(self):
    #     yolo_explorer = YoloPersonExplorer()
    #     user_id = 73636
    #
    #     yolo_explorer.predict_and_save(
    #                                    'https://youtu.be/6yYchgX1fMw?si=OsEIsqdIPoyzST8R',
    #                                    get_predict_path_by_user(user_id))
    #
    #     result_path = os.path.join(Settings.predict_dir, str(user_id) + '/crops/person')
    #
    #     self.assertTrue(len(os.listdir(result_path)) == Settings.max_crop)
    #
    #     # Delete a non-empty directory called 'thedirectory'
    #     shutil.rmtree(os.path.join(Settings.predict_dir, str(user_id)))
