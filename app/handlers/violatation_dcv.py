import shutil

from app.handlers.yolo_explorer import YoloPersonExplorer
from app.handlers.violation_detection import ViolationDetector


import os
from app.utils.filepath_editor import get_predict_path, get_images_path, get_img_path


class ViolatationDCV:
    def __init__(self):
        self.person_explorer = YoloPersonExplorer()
        self.violation_detector = ViolationDetector()

    def detect(self, data, project_path):
        self.person_explorer.predict_and_save(data, project_path)

        # Check whether the specified path exists or not
        isExist = os.path.exists(project_path+'/done')
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(project_path+'/done')

        predict_path = get_predict_path(project_path)

        images = os.listdir(get_images_path(predict_path))

        results = list()
        for img in images:
            print(img)
            t_img = get_img_path(predict_path, img)
            predict = self.violation_detector.predict(t_img)

            done_img = project_path+'/done/'+img
            shutil.move(t_img, done_img)

            results.append({'img': t_img, 'label': predict})

            if os.path.isdir(predict_path):
                shutil.rmtree(predict_path)

        return results
