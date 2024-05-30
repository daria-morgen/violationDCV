from ultralytics import YOLO


class YOLOTrainer(object):
    def __init__(self, args):
        self.model = YOLO(args.yolo_model_dir)


    def train(self):
        # Training.
        results = self.model.train(
            data='/Users/Daria/projects/PycharmProjects/violationDCV/app/datasets/yolo/railway_markings_2.v1i.yolov8/data.yaml',
            imgsz=1280,
            epochs=50,
            batch=8,
            name='yolov8n_v8_50e'
        )

        return results


