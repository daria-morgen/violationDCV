import shutil

from app.handlers.yolo_explorer import YoloPersonExplorer
from app.handlers.violation_detection_explorer import ViolationDetector

import os


class ViolationDCV:
    def __init__(self, args):
        self.person_explorer = YoloPersonExplorer(args)
        self.violation_detector = ViolationDetector(args)

    def clear(self, project):

        predicts = project + '/predict'

        if os.path.isdir(project + '/predict'):
            shutil.rmtree(predicts)

    @staticmethod
    def write_log(project, img, label, status, description):
        with open(project + '/results.txt', 'a') as file:
            file.write(
                "{'img': '" + img + "', 'label': '" + label + "', 'status': '" + status + "', 'description: '" + description +
                "'}\n")

    def detect(self, data, project):

        global pe

        if not os.path.isdir(project + '/images/'):
            os.mkdir(project + '/images/')
            os.mkdir(project + '/done_crops/')

        data = shutil.copy(data, project + '/images')

        results = list()

        try:
            pe = self.person_explorer.predict_and_save(data, project)
        except Exception as e:
            self.write_log(project, data, '', 'error', str(e))
            self.clear(project)
            raise

        if len(pe[0].boxes) > 0:
            predict_person_crops = os.path.join(project, 'predict/crops/person')
            if os.path.isdir(predict_person_crops):
                images = os.listdir(predict_person_crops)

                for img in images:
                    t_img = os.path.join(predict_person_crops, img)

                    try:
                        predict = self.violation_detector.predict(t_img)

                        done_img = project + '/done_crops/' + img
                        shutil.move(t_img, done_img)
                        results.append({'img': t_img, 'label': predict})
                        self.write_log(project, done_img, predict, 'success', '')

                    except Exception as e:
                        self.write_log(project, data, '', 'error', str(e))
            else:
                self.write_log(project, data, '', 'done', 'No persons found')

        self.write_log(project, data, '', 'done', '')
        self.clear(project)

        return results
