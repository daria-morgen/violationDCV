from app.config.settings import Settings
from app.handlers.yolo_explorer import YoloPersonExplorer
from app.handlers.violation_detection import ViolationDetector

import os
from app.utils.filepath_editor import get_crop_path, get_img_path


class ViolatationDCV:
    def __init__(self):
        self.person_explorer = YoloPersonExplorer()
        self.violation_detector = ViolationDetector()

    def detect(self, data, project_path):

        self.person_explorer.predict_and_save(data, project_path)

        crop_path = get_crop_path(project_path)

        images = os.listdir(crop_path)

        results = list()
        for img in images:
            t_img = get_img_path(crop_path, img)
            predict = self.violation_detector.predict(t_img)
            results.append({'img': t_img, 'label': predict})

        return results