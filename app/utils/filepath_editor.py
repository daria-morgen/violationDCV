import os


def get_img_path(predict_path, img):
    img_path = os.path.join(predict_path, 'crops/person/' + img)
    return img_path


def get_images_path(predict_path):
    img_path = os.path.join(predict_path, 'crops/person/')
    return img_path


def get_predict_path(project_path):
    images_path = os.path.join(project_path, 'predict/')
    return images_path


