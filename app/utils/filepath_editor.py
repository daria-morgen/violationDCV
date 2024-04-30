import os
from app.config.settings import Settings


def get_img_path(crop_path, img):
    img_path = os.path.join(crop_path, img)
    return img_path


def get_crop_path(project_path):
    images_path = os.path.join(project_path, 'predict/crops/person')
    return images_path


def get_predict_path_by_user(user_id):
    corp_path = os.path.join(Settings.predict_dir, str(user_id))
    return corp_path


def get_crop_path_by_user(user_id):
    images_path = os.path.join(get_predict_path_by_user(user_id), 'predict/crops/person')
    return images_path
