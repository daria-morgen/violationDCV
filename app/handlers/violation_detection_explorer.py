import torch
import torch.nn.functional as F

from app.trainers.vdcv_trainer import ConvNet
from app.utils import image_formatter


def okay():
    return "okay"


def bad():
    return "bad"


def unknown():
    return "unknown"


class ViolationDetector:
    def __init__(self, args):
        self.model = ConvNet()
        self.model.load_state_dict(torch.load(args.vdcv_model_dir))

    def predict(self, img):
        pred = self.model(image_formatter.format_img(img).unsqueeze(0))

        probs = F.softmax(pred.detach()).numpy().argmax(1)

        switch = {
            0: okay(),
            1: bad(),
            2: unknown()
        }

        return switch.get(probs[0])
