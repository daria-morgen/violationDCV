from ultralytics import YOLO
from app.config.settings import Settings


class YoloPersonExplorer:

    def __init__(self):
        self.model = YOLO(Settings.yolo_model_dir)

    def predict_and_save(self, data, project_path):

        results = self.model.predict(data,
                           save_crop=True, #save_txt=True,
                           imgsz=320, conf=0.5,
                           project=project_path,
                           classes=[0], vid_stride=50)

        assert (len(results[0].boxes) != 0), "No persons found"

