import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
sys.path.append(".")
sys.path.append("..")
import torch
import torch.distributed as dist

from configs.config import Config
from train_net import train_net
import torch.multiprocessing as mp


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)


def init():
    par_dir = os.path.dirname(os.getcwd())
    print(par_dir)
    cfg = Config('volleyball')
    cfg.data_path = os.path.join(par_dir, 'data/volleyball')
    cfg.use_gpu = True
    # cfg.use_multi_gpu=True
    cfg.training_stage = 1
    cfg.stage1_model_path = ''
    cfg.train_backbone = True
    cfg.test_before_train = False

    # VGG16
    cfg.backbone = 'res18'
    cfg.image_size = 720, 1280
    cfg.out_size = 23, 40
    cfg.emb_features = 512

    cfg.num_before = 5
    cfg.num_after = 4

    cfg.batch_size = 2
    cfg.test_batch_size = 1
    cfg.num_frames = 1
    # cfg.train_learning_rate=1e-5
    # cfg.lr_plan={}
    # cfg.max_epoch=200
    cfg.train_learning_rate = 1e-4
    cfg.lr_plan = {30: 5e-5, 60: 2e-5, 90: 1e-5}
    cfg.max_epoch = 120
    cfg.set_bn_eval = False
    cfg.actions_weights = [[1., 1., 2., 3., 1., 2., 2., 0.2, 1.]]

    cfg.exp_note = 'Volleyball_stage1'
    return cfg


def main(rank, world_si):
    cfg = init()
    ddp_setup(rank, world_si)
    train_net(cfg)


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    print("world size:%d" % world_size)
    # main()
    mp.spawn(main, args=(world_size,), nprocs=world_size)
