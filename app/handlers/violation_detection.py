import torch
import torch.nn.functional as F


from app.config.settings import Settings
from train.main import ConvNet
from app.utils import image_formatter


class ViolationDetector:
    def __init__(self):
        self.model = ConvNet()
        self.model.load_state_dict(torch.load(Settings.model_dir))

    def predict(self, img_path):

        pred = self.model(image_formatter.format_img(img_path).unsqueeze(0))

        probs = F.softmax(pred.detach()).numpy().argmax(1)

        return probs
