import os
from unittest import TestCase

from app.handlers.yolo_explorer import YoloPersonExplorer
import shutil

from app.options.train_options import TrainCompOptions


class TestYoloPersonExplorer(TestCase):
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

    def test_predict_and_save_create_user_dir(self):
        yolo_explorer = YoloPersonExplorer(self.args)
        img = './data/okey470.jpg'

        yolo_explorer.predict_and_save(img,
                                       self.project)

        self.assertTrue(os.path.isdir('./data/project_predicts/predict'))


    def test_predict_and_save_predict_person(self):
        yolo_explorer = YoloPersonExplorer(self.args)
        img = './data/okey470.jpg'

        yolo_explorer.predict_and_save(
            img,
            self.project)

        self.assertTrue(len('./data/project_predicts/predict/crops/person') > 0)

        # Delete a non-empty directory called 'thedirectory'

    def test_predict_and_save_user_person_not_found(self):
        yolo_explorer = YoloPersonExplorer(self.args)
        img = './data/unknow88.jpg'

        with self.assertRaises(Exception):
            yolo_explorer.predict_and_save(img,
                                           self.project)


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
