from app.config.settings import Settings
from app.handlers.yolo_explorer import YoloHumanExplorer
from app.handlers.violation_detection import ViolationDetector

import os


class ViolatationDCV:
    def __init__(self):
        self.human_explorer = YoloHumanExplorer()
        self.violation_detector = ViolationDetector()

    def detect(self, user_id, img_path):
        self.human_explorer.predict_and_save(1, img_path)

        crop_path = os.path.join(Settings.predict_dir, str(user_id))

        assert (os.path.isdir(crop_path) != 0), "No humans found"

