import json
import pickle
import numpy as np
import skimage.io
import skimage.transform

import torch
import torchvision.transforms as transforms
from torch.utils import data
import torchvision.models as models
from collections import defaultdict
from PIL import Image
import random

import sys

"""
Reference:
https://github.com/cvlab-epfl/social-scene-understanding/blob/master/volleyball.py
"""

ACTIVITIES = ['r_set', 'r_spike', 'r-pass', 'r_winpoint',
              'l_set', 'l-spike', 'l-pass', 'l_winpoint']

NUM_ACTIVITIES = 8

ACTIONS = ['blocking', 'digging', 'falling', 'jumping',
           'moving', 'setting', 'spiking', 'standing',
           'waiting']
NUM_ACTIONS = 9


def volley_read_annotations(path):
    """
    reading annotations for the given sequence
    """
    annotations = {}
    # build the mapping from act to id 生成动作和id的键值对字典{动作 ： id}
    gact_to_id = {name: i for i, name in enumerate(ACTIVITIES)} # 群体
    act_to_id = {name: i for i, name in enumerate(ACTIONS)} # 个体

    with open(path) as f: # 打开标注文件
        # 对每一帧的标注
        for l in f.readlines():
            # 每一行数据存在value里
            values = l[:-1].split(' ')     # 因为最后一个为换行符，所以去掉，values列表存储一系列值
            file_name = values[0]
            activity = gact_to_id[values[1]]

            # 个体bounding box以及类别标注
            values = values[2:]
            num_people = len(values) // 5
            # store all the actions in the scene
            action_names = values[4::5]
            actions = [act_to_id[name]
                       for name in action_names]

            def _read_bbox(xywh): # 将xyhw转化成左下和右上的两对坐标数
                x, y, w, h = map(int, xywh)
                return y, x, y + h, x + w

            # 每个人的边界框存在bboxes中 , bbox's shape (12,4)(11,4)(10,4)均可能
            bboxes = np.array([_read_bbox(values[i:i + 4])
                               for i in range(0, 5 * num_people, 5)])

            # 每个标注行的文件id,实际上就是clip id, 对应的图像序列在标注文件同级下的相同id的目录下
            fid = int(file_name.split('.')[0])
            annotations[fid] = {
                'file_name': file_name,
                'group_activity': activity,
                'actions': actions,
                'bboxes': bboxes,
            }
            # 综上，对于输出的标注(返回的字典annotations)可以表示为形如：
            """
            {
                1:{ # fid=1（来自"0001.jpg"）
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
            """
    return annotations

def volleyball_readpose(data_path):
    f = open(data_path,'r')
    f = f.readlines()
    pose_ann= defaultdict(dict)
    for ann in f:
        values = ann.split(' ')
        pickle_path = values[0]
        with open(pickle_path,'rb') as f:
            keypoints = pickle.load(f)
        sid = int(values[1])
        src_id = int(values[2])
        pose_ann[sid][src_id] = keypoints
    return pose_ann

def volley_read_dataset(path, seqs):
    data = {}
    # 按照视频序列进行循环，集合每段视频（seq）序列的标注，每段视频里面的标注又由多个帧的标注组成，再嵌套一层
    for sid in seqs:
        # 对每个seq
        data[sid] = volley_read_annotations(path + '/%d/annotations.txt' % sid) # 插入键值对 seq : 标签
    return data


def volley_all_frames(data):
    frames = []
    for sid, anns in data.items():
        for fid, ann in anns.items():
            frames.append((sid, fid))
    return frames


def volley_random_frames(data, num_frames):
    frames = []
    for sid in np.random.choice(list(data.keys()), num_frames):
        fid = int(np.random.choice(list(data[sid]), []))
        frames.append((sid, fid))
    return frames


def volley_frames_around(frame, num_before=5, num_after=4):
    sid, src_fid = frame
    return [(sid, src_fid, fid)
            for fid in range(src_fid - num_before, src_fid + num_after + 1)]


def load_samples_sequence(anns, tracks, images_path, frames, image_size, num_boxes=12, ):
    """
    load samples of a bath
    
    Returns:
        pytorch tensors
    """
    images, boxes, boxes_idx = [], [], []
    activities, actions = [], []
    for i, (sid, src_fid, fid) in enumerate(frames):
        # img=skimage.io.imread(images_path + '/%d/%d/%d.jpg' % (sid, src_fid, fid))
        # img=skimage.transform.resize(img,(720, 1280),anti_aliasing=True)

        img = Image.open(images_path + '/%d/%d/%d.jpg' % (sid, src_fid, fid))

        img = transforms.functional.resize(img, image_size)
        img = np.array(img)

        # H,W,3 -> 3,H,W
        img = img.transpose(2, 0, 1)
        images.append(img)

        boxes.append(tracks[(sid, src_fid)][fid])
        actions.append(anns[sid][src_fid]['actions'])
        if len(boxes[-1]) != num_boxes:
            boxes[-1] = np.vstack([boxes[-1], boxes[-1][:num_boxes - len(boxes[-1])]])
            actions[-1] = actions[-1] + actions[-1][:num_boxes - len(actions[-1])]
        boxes_idx.append(i * np.ones(num_boxes, dtype=np.int32))
        activities.append(anns[sid][src_fid]['group_activity'])

    images = np.stack(images)
    activities = np.array(activities, dtype=np.int32)
    bboxes = np.vstack(boxes).reshape([-1, num_boxes, 4])
    bboxes_idx = np.hstack(boxes_idx).reshape([-1, num_boxes])
    actions = np.hstack(actions).reshape([-1, num_boxes])

    # convert to pytorch tensor
    images = torch.from_numpy(images).float()
    bboxes = torch.from_numpy(bboxes).float()
    bboxes_idx = torch.from_numpy(bboxes_idx).int()
    actions = torch.from_numpy(actions).long()
    activities = torch.from_numpy(activities).long()

    return images, bboxes, bboxes_idx, actions, activities


class VolleyballDataset(data.Dataset):
    """
    Characterize volleyball dataload for pytorch
    """

    def __init__(self, anns, tracks, frames, images_path, keypoints, image_size, feature_size, inference_module_name, num_boxes=12,
                 num_before=4, num_after=4, is_training=True, is_finetune=False):
        self.anns = anns
        self.tracks = tracks
        self.frames = frames
        # 数据集路径
        self.images_path = images_path
        # 图像大小
        self.image_size = image_size
        # backbone输出特征图大小
        self.feature_size = feature_size
        # dynamic_volleyball
        self.inference_module_name = inference_module_name
        self.pose_anns=volleyball_readpose(keypoints)

        self.num_boxes = num_boxes
        self.num_before = num_before
        self.num_after = num_after

        self.is_training = is_training
        self.is_finetune = is_finetune
        self.idx2class = {
                    0: 'r_set',
                    1: 'l_set', 
                    2: 'r_spike',
                    3: 'l_spike',
                    4: 'r_pass',
                    5: 'l_pass',
                    6: 'r_winpoint',
                    7: 'l_winpoint'
        }
        self.class2idx = dict()
        for k in self.idx2class:
            self.class2idx[self.idx2class[k]] = k
        # self.frames_seq = np.empty((1337, 2), dtype = np.int)
        # self.flag = 0

    def __len__(self):
        """
        Return the total number of samples
        所有剪辑的数量
        """
        return len(self.frames)

    def __getitem__(self, index):
        """
        Generate one sample of the dataload
        """
        # Save frame sequences
        # self.frames_seq[self.flag] = self.frames[index]# [0], self.frames[index][1]
        # if self.flag == 1336:
        #     save_seq = self.frames_seq
        #     np.savetxt('vis/frames_seq.txt', save_seq)
        # self.flag += 1

        # 首先进行采样, 得到三元组列表
        sid,fid = self.frames[index]
        select_frames = self.volley_frames_sample(self.frames[index])
        # 载入选择的采用帧, 得到 tuple (images, bboxes, actions, activities)
        images, bboxes, actions, activities, poses = self.load_samples_sequence(select_frames)
        return images, bboxes, actions, activities, poses,sid,fid

    def volley_frames_sample(self, frame):
        # return a list of a tuple (sid, src_fid, fid)
        sid, src_fid = frame

        if self.is_finetune:
            if self.is_training:
                # 如果是训练的话【】里只有一帧，即采样帧
                fid = random.randint(src_fid - self.num_before, src_fid + self.num_after)
                return [(sid, src_fid, fid)]
            else:
                # 如果是推理的话， 【】存的是剪辑里的一定范围的所有帧
                return [(sid, src_fid, fid)
                        for fid in range(src_fid - self.num_before, src_fid + self.num_after + 1)]
        else:
            # if self.is_training:
            #     sample_frames=random.sample(range(src_fid-self.num_before, src_fid+self.num_after+1), 3)
            #     return [(sid, src_fid, fid)
            #             for fid in sample_frames]
            # else:
            #     return [(sid, src_fid, fid)
            #             for fid in  [src_fid-3,src_fid,src_fid+3, src_fid-4,src_fid-1,src_fid+2, src_fid-2,src_fid+1,src_fid+4 ]]
            if self.inference_module_name == 'arg_volleyball':
                if self.is_training:
                    sample_frames = random.sample(range(src_fid - self.num_before, src_fid + self.num_after + 1), 3)
                    return [(sid, src_fid, fid)
                            for fid in sample_frames]
                else:
                    return [(sid, src_fid, fid)
                            for fid in
                            [src_fid - 3, src_fid, src_fid + 3, src_fid - 4, src_fid - 1, src_fid + 2, src_fid - 2,
                             src_fid + 1, src_fid + 4]]
            else:
                # 如果不是arg_volleyball推理模块, 使用指定范围的全部帧
                # if self.is_training:
                return [(sid, src_fid, fid) for fid in
                        range(src_fid - self.num_before, src_fid + self.num_after + 1)]
            # else:
            #     return [(sid, src_fid, fid) for fid in
            #             range(src_fid - self.num_before, src_fid + self.num_after + 1)]

    def load_samples_sequence(self, select_frames):
        """
        load samples sequence

        Returns:
            pytorch tensors
        """

        OH, OW = self.feature_size

        # 所有样本images存在images中
        images, boxes = [], []
        activities, actions = [], []
        poses = []
        for i, (sid, src_fid, fid) in enumerate(select_frames):

            img = Image.open(self.images_path + '/%d/%d/%d.jpg' % (sid, src_fid, fid))

            # 对image的大小进行转换
            img = transforms.functional.resize(img, self.image_size)
            img = np.array(img)

            # H,W,3 -> 3,H,W
            img = img.transpose(2, 0, 1)
            images.append(img)

            temp_boxes = np.ones_like(self.tracks[(sid, src_fid)][fid])
            temp_poses = []
            keypoints = self.pose_anns[sid][src_fid][fid]
            for i, track in enumerate(self.tracks[(sid, src_fid)][fid]):
                y1, x1, y2, x2 = track
                # multiply output channel shape
                w1, h1, w2, h2 = x1 * OW, y1 * OH, x2 * OW, y2 * OH
                temp_boxes[i] = np.array([w1, h1, w2, h2])
                temp_poses.append(keypoints[i])
            
            if len(temp_poses) != self.num_boxes:
                temp_poses = temp_poses + temp_poses[:self.num_boxes -len(temp_poses)]
            temp_poses = np.vstack(temp_poses)


            boxes.append(temp_boxes)
            poses.append(temp_poses)

            actions.append(self.anns[sid][src_fid]['actions'])

            if len(boxes[-1]) != self.num_boxes:
                # if not equal , copy the missing ones to make it equal
                boxes[-1] = np.vstack([boxes[-1], boxes[-1][:self.num_boxes - len(boxes[-1])]])
                actions[-1] = actions[-1] + actions[-1][:self.num_boxes - len(actions[-1])]

            activities.append(self.anns[sid][src_fid]['group_activity'])

        images = np.stack(images)  # T,3,H,W
        activities = np.array(activities, dtype=np.int32)
        # print(f"pre bboxes shape: {np.vstack(boxes).shape}")
        bboxes = np.vstack(boxes).reshape([-1, self.num_boxes, 4]) # T,12,4
        poses = np.vstack(poses).reshape([-1,self.num_boxes, 17,3])

        # print(f"poses shape: {poses.shape}")
        # print(f"count:{self.get_count()}")
        actions = np.hstack(actions).reshape([-1, self.num_boxes]) # T,12

        # convert to pytorch tensor
        images = torch.from_numpy(images).float()
        bboxes = torch.from_numpy(bboxes).float()
        poses = torch.from_numpy(poses).float()
        actions = torch.from_numpy(actions).long()
        activities = torch.from_numpy(activities).long() # T,

        return images, bboxes, actions, activities, poses
