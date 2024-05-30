import argparse
import os
import torch
from mmcv.runner import init_dist, get_dist_info
import torch.distributed as dist


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--name', type=str, default="test", help='Name of this trial')
        self.parser.add_argument('--decomp_name', type=str, default="Decomp_SP001_SM001_H512", help='Name of autoencoder model')
        self.parser.add_argument("--gpu_id", type=int, nargs='+', default=(-1), help='GPU id')
        self.parser.add_argument("--distributed", action="store_true", help='Whether to use DDP training')
        self.parser.add_argument("--data_parallel", action="store_true", help="Whether to use DP training")
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--model_dir', type=str, default='/Users/Daria/projects/PycharmProjects/violationDCV/app/handlers/models', help='models are saved here')
        self.parser.add_argument('--yolo_model_dir', type=str, default='/Users/Daria/projects/PycharmProjects/violationDCV/app/handlers/models/yolov8n.pt', help='models are saved here')
        self.parser.add_argument('--vdcv_model_dir', type=str, default='/Users/Daria/projects/PycharmProjects/violationDCV/app/handlers/models/ppe_net.pth', help='models are saved here')
        self.parser.add_argument('--dataset_name', type=str, default='t2m', help='Dataset Name')



        self.parser.add_argument('--train_okay', type=str, default='/Users/Daria/projects/PycharmProjects/violationDCV/app/datasets/vdcv/train/okey', help='models are saved here')
        self.parser.add_argument('--train_bad', type=str, default='/Users/Daria/projects/PycharmProjects/violationDCV/app/datasets/vdcv/train/bad', help='models are saved here')
        self.parser.add_argument('--train_unknow', type=str, default='/Users/Daria/projects/PycharmProjects/violationDCV/app/datasets/vdcv/train/unknow', help='models are saved here')

        self.parser.add_argument('--test_okay', type=str, default='/Users/Daria/projects/PycharmProjects/violationDCV/app/datasets/vdcv/train/okey', help='models are saved here')
        self.parser.add_argument('--test_bad', type=str, default='/Users/Daria/projects/PycharmProjects/violationDCV/app/datasets/vdcv/train/bad', help='models are saved here')
        self.parser.add_argument('--test_unknow', type=str, default='/Users/Daria/projects/PycharmProjects/violationDCV/app/datasets/vdcv/train/unknow', help='models are saved here')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()

        self.opt.is_train = self.is_train

        args = vars(self.opt)
        if args["distributed"]:
            init_dist('slurm')
        rank, world_size = get_dist_info()
        if args["distributed"]:
            self.opt.gpu_id = range(world_size)
        elif self.opt.gpu_id != (-1):
            if len(self.opt.gpu_id) == 1:
                torch.cuda.set_device(self.opt.gpu_id[0])
        else:
            assert args["data_parallel"] == False

        if rank == 0:
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')
            if self.is_train:
                # save to the disk
                expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.dataset_name, self.opt.name)
                if not os.path.exists(expr_dir):
                    os.makedirs(expr_dir)
                file_name = os.path.join(expr_dir, 'opt.txt')
                with open(file_name, 'wt') as opt_file:
                    opt_file.write('------------ Options -------------\n')
                    for k, v in sorted(args.items()):
                        opt_file.write('%s: %s\n' % (str(k), str(v)))
                    opt_file.write('-------------- End ----------------\n')
        if world_size > 1:
            dist.barrier()
        return self.opt