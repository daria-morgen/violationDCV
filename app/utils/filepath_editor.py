import os
from app.config.settings import Settings


def get_img_path(predict_path, img):
    img_path = os.path.join(predict_path, 'crops/person/' + img)
    return img_path


def get_images_path(predict_path):
    img_path = os.path.join(predict_path, 'crops/person/')
    return img_path


def get_predict_path(project_path):
    images_path = os.path.join(project_path, 'predict/')
    return images_path


def get_predict_path_by_user(user_id):
    corp_path = os.path.join(Settings.project_predicts, str(user_id))
    return corp_path


def get_crop_path_by_user(user_id):
    images_path = os.path.join(get_predict_path_by_user(user_id), 'predict/crops/person')
    return images_path
