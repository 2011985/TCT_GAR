import numpy as np
import skimage.io
import skimage.transform

import torch
import torchvision.transforms as transforms
from torch.utils import data
import torchvision.models as models
from numpy.random import randint
from PIL import Image
import random
import os
import sys

"""
Reference:
https://github.com/cvlab-epfl/social-scene-understanding/blob/master/volleyball.py
"""

ACTIVITIES = ['2p-succ.','2p-fail.-off.', '2p-fail.-def.', '2p-layup-succ.', '2p-layup-fail.-off.', '2p-layup-fail.-def.',
              '3p-succ.', '3p-fail.-off.', '3p-fail.-def.']

NUM_ACTIVITIES = 9



def NBA_read_annotations(path, video_dir):
    """
    reading annotations for the given sequence
    """
    annotations = {}
    # build the mapping from act to id
    gact_to_id = {name: i for i, name in enumerate(ACTIVITIES)}

    with open(path) as f:
        for l in f.readlines():
            values = l[:-1].split()
            clip_id = values[0]
            activity_id = gact_to_id[values[1]]
            clip_dir = video_dir+clip_id
            dir_files = os.listdir(clip_dir)
            num_frames = len(dir_files)

            clip_id = int(clip_id)
            annotations[clip_id] = {
                'group_activity': activity_id,
                'num_frames': num_frames
            }
    return annotations


def NBA_read_dataset(path, seqs):
    data = {}
    for sid in seqs:
        data[sid] = NBA_read_annotations(path + '/%d/annotations.txt' % sid, path+'/%d/'%sid)
    return data


def NBA_all_frames(data):
    frames = []
    for sid, anns in data.items():
        for fid, ann in anns.items():
            frames.append((sid, fid))
    return frames


class NBADataset(data.Dataset):
    """
    Characterize NBA dataload for pytorch
    """

    def __init__(self, anns, tracks, frames, images_path, image_size, feature_size, inference_module_name, num_boxes=14, num_segments=8,
                 new_length=1, is_training=True, is_finetune=False):
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

        self.num_boxes = num_boxes
        self.num_segments = num_segments
        self.new_length = new_length
        self.is_training = is_training
        self.is_finetune = is_finetune

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
        # 首先进行采样, 得到offsets
        sid, cid, offsets = self.NBA_frames_sample(self.frames[index])
        # 载入选择的采用帧, 得到 tuple (images, bboxes, activities)
        sample = self.load_samples_sequence(sid,cid,offsets)
        return sample

    def _sample_indices(self, sid,cid):
        """

        :param sid,cid
        :return: list
        """
        num_frames = self.anns[sid][cid]['num_frames']
        average_duration = (num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                              size=self.num_segments)
        elif num_frames > self.num_segments:
            offsets = np.sort(randint(num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets


    def NBA_frames_sample(self, frame):
        # return a list of a tuple (sid, cid, filename)
        sid, cid = frame

        if self.is_finetune:
            offsets = self._sample_indices(sid,cid)
            return sid, cid, offsets
        else:
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

    def load_samples_sequence(self, sid, cid, offsets):
        """
        load samples sequence

        Returns:
            pytorch tensors
        """

        OH, OW = self.feature_size
        num_frames = self.anns[sid][cid]['num_frames']
        images, boxes = [], []
        activities= []

        for offset in offsets:
            off = int(offset)
            for i in range(self.new_length):
                img = Image.open(self.images_path + '/%d/%d/%06d.jpg' % (sid, cid, off))
                img = transforms.functional.resize(img,self.image_size)
                img = np.array(img)

                # H,W,3 -> 3,H,W
                img = img.transpose(2, 0, 1)
                images.append(img)

                temp_boxes = np.ones_like(self.tracks[(sid, cid)][off][:,:4])
                assert(temp_boxes.shape[0]!=0)
                for i, track in enumerate(self.tracks[(sid, cid)][off]):
                    y1, x1, y2, x2, t0 = track
                    # multiply output channel shape
                    w1, h1, w2, h2 = x1 * OW, y1 * OH, x2 * OW, y2 * OH
                    temp_boxes[i] = np.array([w1, h1, w2, h2])

                if temp_boxes.shape[0]>self.num_boxes:
                    boxes.append(temp_boxes[:self.num_boxes])
                else:
                    boxes.append(temp_boxes)

                while len(boxes[-1]) != self.num_boxes:
                    # if not equal , copy the missing ones to make it equal
                    boxes[-1] = np.vstack([boxes[-1], boxes[-1][:self.num_boxes - len(boxes[-1])]])

                activities.append(self.anns[sid][cid]['group_activity'])

                if off < num_frames:
                    off+=1

        images = np.stack(images)  # T,3,H,W
        activities = np.array(activities, dtype=np.int32)
        bboxes = np.vstack(boxes).reshape([-1, self.num_boxes, 4]) # T,14,4

        # convert to pytorch tensor
        images = torch.from_numpy(images).float()
        bboxes = torch.from_numpy(bboxes).float()
        activities = torch.from_numpy(activities).long() # T,

        assert bboxes.shape[0] == 8



        return images, bboxes, activities
