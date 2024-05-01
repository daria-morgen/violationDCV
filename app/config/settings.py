class Settings:
    # user
    parent_dir = "/Users/Daria/projects/PycharmProjects/violationDCV"

    project_predicts = "/Users/Daria/projects/PycharmProjects/violationDCV/project_predicts"
    project_predicts_data = project_predicts+"/data"
    log_file = project_predicts+"/images"

    model_dir = "/Users/Daria/projects/PycharmProjects/violationDCV/app/handlers/models/ppe_net.pth"
    yolo_model_dir = "/Users/Daria/projects/PycharmProjects/violationDCV/app/handlers/models/yolov8n.pt"

    max_crop = 10

    train_okey_path = '/Users/Daria/projects/PycharmProjects/violationDCV/train/darasets/train/okey'
    train_bad_path = '/Users/Daria/projects/PycharmProjects/violationDCV/train/darasets/train/bad'
    train_unknow_path = '/Users/Daria/projects/PycharmProjects/violationDCV/train/darasets/train/unknow'

    test_okey_path = '/Users/Daria/projects/PycharmProjects/violationDCV/train/darasets/test/okey'
    test_bad_path = '/Users/Daria/projects/PycharmProjects/violationDCV/train/darasets/test/bad'
    test_unknow_path = ('/Users/Daria/projects/PycharmProjects/violationDCV/train/darasets/test'
                        '/unknow')
