import shutil

from app.handlers.yolo_explorer import YoloPersonExplorer
from app.handlers.violation_detection import ViolationDetector

import os
from app.utils.filepath_editor import get_predict_path, get_images_path, get_img_path


class ViolatationDCV:
    def __init__(self):
        self.person_explorer = YoloPersonExplorer()
        self.violation_detector = ViolationDetector()

    def clear(self, project_path):

        predicts = project_path + '/predict'

        if os.path.isdir(project_path + '/predict'):
            shutil.rmtree(predicts)

    @staticmethod
    def write_log(project_path, img, label, status, description):
        with open(project_path + '/results.txt', 'a') as file:
            file.write(
                "{'img': '" + img + "', 'label': '" + label + "', 'status': '" + status + "', 'description: '" + description +
                "'}\n")

    def detect(self, data, project_path):

        global pe
        data = shutil.copy(data, project_path + '/images')

        results = list()

        try:
            pe = self.person_explorer.predict_and_save(data, project_path)
        except Exception as e:
            self.write_log(project_path, data, '', 'error', str(e))
            self.clear(project_path)

        if len(pe[0].boxes) > 0:
            predict_person_crops = os.path.join(project_path, 'predict/crops/person')

            images = os.listdir(predict_person_crops)

            for img in images:
                t_img = os.path.join(predict_person_crops, img)

                try:
                    predict = self.violation_detector.predict(t_img)

                    done_img = project_path + '/done_crops/' + img
                    shutil.move(t_img, done_img)
                    results.append({'img': t_img, 'label': predict})
                    self.write_log(project_path, done_img, predict, 'success', '')

                except Exception as e:
                    self.write_log(project_path, data, '', 'error', str(e))

        self.write_log(project_path, data, '', 'done', '')
        self.clear(project_path)

        return results
