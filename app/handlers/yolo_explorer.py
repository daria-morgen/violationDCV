from ultralytics import YOLO

import os
from app.config.settings import Settings


class YoloHumanExplorer:

    def __init__(self):
        self.model = YOLO('yolov8n.pt')

    def predict_and_save(self, user_id, data):

        self.model.predict(data,
                           save_crop=True,
                           imgsz=320, conf=0.5,
                           project=os.path.join(Settings.predict_dir, str(user_id)),
                           classes=[0], vid_stride=50)

