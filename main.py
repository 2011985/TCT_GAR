import os

os.environ['CUDA_VISIBLE_DEVICES'] = "4"

import torch
import torch.distributed as dist

from train_net import train_net
import torch.multiprocessing as mp
from configs import get_config
import torch.profiler


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12357"
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)


def main(rank, world_si, my_config):
    cfg = get_config(my_config)
    ddp_setup(rank, world_si)
    train_net(cfg)


if __name__ == '__main__':
    import sys

    fig = sys.argv[1]
    if not fig:
        print("Need a config name!")
        exit()
    world_size = torch.cuda.device_count() # 返回一个整数，表示你的机器上有多少块 NVIDIA GPU 可以被 PyTorch 使用，返回1说明足够
    print("world size:%d" % world_size)
    # main()
    mp.spawn(main, args=(world_size, fig), nprocs=world_size)
