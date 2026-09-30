import numpy
import torch.utils.data

from dataload.volleyball import *
from dataload.collective import *
from dataload.NBA import *
from torch.utils.data.distributed import DistributedSampler
import pickle


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def return_dataloader(cfg):
    if cfg.dataset_name == 'volleyball':
        # train_seqs 是预定义好的训练的视频序列
        # self.train_seqs = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53,
        #                   54, 0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]  # video id list of train set
        # self.test_seqs = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]  # video id list of test set
        # train_anns 是一个字典，key为视频id，value是一个字典，其key是clip id ，值是一个字典:
        #           {     'file_name': file_name,
        #                 'group_activity': activity,
        #                 'actions': actions,
        #                 'bboxes': bboxes  }
        # 对于train_anns字典可以表示为形如：
        """{
             1：# 对于视频序列1
                {
                    1:{ # fid=1（来自"0001.jpg"）对于序列1中的第1帧图片
                        'file_name': '0001.jpg',
                        'group_activity': 0,  # "attack"→0
                        'actions': [0, 1],    # "spike"→0, "block"→1
                        'bboxes': np.array([
                            [200, 100, 260, 150],  # 人物1: y_min=200, x_min=100, y_max=260, x_max=150
                            [160, 150, 210, 190]   # 人物2
                        ])
                    }
                    2:...
                }
              3：#对于视频序列3
                ...
            }
        """
        train_anns = volley_read_dataset(cfg.data_path, cfg.train_seqs)
        # train_frames stores the list of all (sid, fid) pairs 由所有（视频序列id, 帧id）元组构成的一个列表
        train_frames = volley_all_frames(train_anns)
        
        test_anns = volley_read_dataset(cfg.data_path, cfg.test_seqs)
        test_frames = volley_all_frames(test_anns)
        # **合并字典
        all_anns = {**train_anns, **test_anns}
        # what tracks are exactly ?
        # 是一个字典，其key为(sid, fid) pairs, 其value是一个字典，其key 为 剪辑中的某一帧id, value为bounding box
        all_tracks = pickle.load(open(cfg.data_path + '/tracks_normalized.pkl', 'rb'))

        # building training and validation dataload
        # 只有第一阶段才进行finetune
        training_set = VolleyballDataset(all_anns, all_tracks, train_frames,
                                         cfg.data_path, cfg.keypoints, cfg.image_size, cfg.out_size, cfg.inference_module_name,
                                         num_before=cfg.num_before,
                                         num_after=cfg.num_after, is_training=True,
                                         is_finetune=(cfg.training_stage == 1))
        # print(f"trainning infer module:", training_set.inference_module_name)
        train_sampler = DistributedSampler(training_set,shuffle=False)

        params = {
            'batch_size': cfg.batch_size,
            'shuffle': False,
            'num_workers': 4,
            'pin_memory': True,
            'sampler': train_sampler
        }
        # **对应传参
        train_loader = torch.utils.data.DataLoader(training_set, **params)
        validation_set = VolleyballDataset(all_anns, all_tracks, test_frames,
                                           cfg.data_path, cfg.keypoints, cfg.image_size, cfg.out_size, cfg.inference_module_name,
                                           num_before=cfg.num_before,
                                           num_after=cfg.num_after, is_training=False,
                                           is_finetune=(cfg.training_stage == 1))
        val_sampler = DistributedSampler(validation_set)
        params['batch_size'] = cfg.test_batch_size
        params['sampler'] = val_sampler
        validation_loader = torch.utils.data.DataLoader(validation_set, **params)
    
    elif cfg.dataset_name == 'NBA':

        train_anns = NBA_read_dataset(cfg.data_path, cfg.train_seqs)
        # train_frames stores the list of all (sid, fid) pairs
        train_frames = NBA_all_frames(train_anns)

        test_anns = NBA_read_dataset(cfg.data_path, cfg.test_seqs)
        test_frames = NBA_all_frames(test_anns)

        all_anns = {**train_anns, **test_anns}
        # what tracks are exactly ?
        # 是一个字典，其key为(sid, fid) pairs, 其value是一个字典，其key 为 剪辑中的某一帧id, value为bounding box
        all_tracks = pickle.load(open(cfg.data_path + '/../tracks_normalized_detected_all.pkl', 'rb'))
        # building training and validation dataload
        # 只有第一阶段才进行finetune
        training_set = NBADataset(all_anns, all_tracks, train_frames,
                                         cfg.data_path, cfg.image_size, cfg.out_size, cfg.inference_module_name,
                                         is_training=True,
                                         is_finetune=(cfg.training_stage == 1))
        # print(f"trainning infer module:", training_set.inference_module_name)
        train_sampler = DistributedSampler(training_set)

        params = {
            'batch_size': cfg.batch_size,
            'shuffle': False,
            'num_workers': 4,
            'pin_memory': True,
            'sampler': train_sampler
        }
        train_loader = torch.utils.data.DataLoader(training_set, **params)
        validation_set = NBADataset(all_anns, all_tracks, test_frames,
                                           cfg.data_path, cfg.image_size, cfg.out_size, cfg.inference_module_name,
                                           is_training=False,
                                           is_finetune=(cfg.training_stage == 1))
        val_sampler = DistributedSampler(validation_set)
        params['batch_size'] = cfg.test_batch_size
        params['sampler'] = val_sampler
        validation_loader = torch.utils.data.DataLoader(validation_set, **params)

    # elif cfg.dataset_name == 'collective':
    #     train_anns = collective_read_dataset(cfg.data_path, cfg.train_seqs)
    #     train_frames = collective_all_frames(train_anns)
    #
    #     test_anns = collective_read_dataset(cfg.data_path, cfg.test_seqs)
    #     test_frames = collective_all_frames(test_anns)
    #
    #     training_set = CollectiveDataset(train_anns, train_frames,
    #                                      cfg.data_path, cfg.image_size, cfg.out_size,
    #                                      num_frames=cfg.num_frames, is_training=True,
    #                                      is_finetune=(cfg.training_stage == 1))
    #
    #     validation_set = CollectiveDataset(test_anns, test_frames,
    #                                        cfg.data_path, cfg.image_size, cfg.out_size,
    #                                        num_frames=cfg.num_frames, is_training=False,
    #                                        is_finetune=(cfg.training_stage == 1))

    else:
        assert False

    # print('Reading dataload finished...')
    # print('%d train samples' % len(train_frames))
    # print('%d test samples' % len(test_frames))

    return train_loader, validation_loader
    # return training_set, validation_set
