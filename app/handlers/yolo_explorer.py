from ultralytics import YOLO


class YoloPersonExplorer:

    def __init__(self, args):
        self.model = YOLO(args.yolo_model_dir)

    def predict_and_save(self, data, project_path):
        results = self.model.predict(data, save=True,
                                     save_crop=True,  #save_txt=True,
                                     imgsz=320, conf=0.5,
                                     project=project_path,
                                     classes=[0], vid_stride=50)

        if len(results[0].boxes).__eq__(0):
            raise "No persons found"

        return results
