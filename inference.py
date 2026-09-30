import os
import random
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from configs import get_config
import torch
from utils import print_log
from PIL import Image
import pickle
import torchvision.transforms as transforms
from model.base_model import Basenet_NBA
from dataload.NBA import ACTIVITIES as NBA_ACTIVITIES
from numpy.random import randint
import numpy as np
from draw_box_utils import draw_objs_NBA

gact_to_id = {name: i for i, name in enumerate(NBA_ACTIVITIES)}
id_to_gact = {i:name for i,name in enumerate(NBA_ACTIVITIES)}
def _sample_indices(num_frames, num_segments):
        """

        :param sid,cid
        :return: list
        """
        average_duration = num_frames // num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(num_segments)), average_duration) + randint(average_duration,
                                                                                              size=num_segments)
        elif num_frames > num_segments:
            offsets = np.sort(randint(num_frames, size=num_segments))
        else:
            offsets = np.zeros((num_segments,))
        return offsets

def append_boxes(cfg, offsets, clip, fea_shape, is_infer):
    all_tracks = pickle.load(open(cfg.data_path + '/../tracks_normalized_detected_all.pkl', 'rb'))
    OH,OW = fea_shape
    boxes = []
    for offset in offsets:
        temp_boxes = np.ones_like(all_tracks[(21801058, clip)][offset][:,:4])

        for i, track in enumerate(all_tracks[(21801058, clip)][offset]):
            y1, x1, y2, x2,_ = track
            # multiply output channel shape
            if is_infer:
                w1, h1, w2, h2 = x1 * OW, y1 * OH, x2 * OW, y2 * OH
            else:
                h1, w1, h2, w2 = x1 * OH, y1 * OW, x2 * OH, y2 * OW

            temp_boxes[i] = np.array([w1, h1, w2, h2])

        if temp_boxes.shape[0]>cfg.num_boxes:
            boxes.append(temp_boxes[:cfg.num_boxes])
        else:
            boxes.append(temp_boxes)

        while len(boxes[-1]) != cfg.num_boxes:
            # if not equal , copy the missing ones to make it equal
            boxes[-1] = np.vstack([boxes[-1], boxes[-1][:cfg.num_boxes - len(boxes[-1])]])
    return boxes



def get_batch_data(cfg, videos_dir, device):

    if not os.path.exists(videos_dir):
        print(" not found {} path".format(videos_dir))
        return None
    label_path = os.path.join(videos_dir, "annotations.txt")
    if not os.path.exists(label_path):
        print(" not found {} file".format(label_path))
        return None

    

    label_info = []

    with open(label_path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            # 每一行数据存在value里
            values = line.split()
            clip = values[0]
            clip_dir = os.path.join(videos_dir,clip)
            dir_files = os.listdir(clip_dir)
            num_frames = len(dir_files)
            activity = gact_to_id[values[1]]
            label_info.append([int(clip), activity, num_frames])

    OH, OW = cfg.out_size
    all_tracks = pickle.load(open(cfg.data_path + '/../tracks_normalized_detected_all.pkl', 'rb'))

    b_videos = []
    b_activities = []
    b_boxes = []
    b_boxes_real = []

    for enti in label_info:

        clip, activity, num_frames = enti
        images, boxes = [], []
        activities = []
        offsets =  _sample_indices(num_frames, cfg.num_segments)

        for offset in offsets:
            img = Image.open(videos_dir + '/%d/%06d.jpg' % (clip, offset))
            # 对image的大小进行转换
            img = transforms.functional.resize(img, cfg.image_size)
            img = np.array(img)
            # H,W,3 -> 3,H,W
            img = img.transpose(2, 0, 1)
            images.append(img)

            activities.append(activity)
        boxes = append_boxes(cfg, offsets, clip, (OH,OW), True)
        fea_shape = (1080,1960)
        boxes_real = append_boxes(cfg,offsets,clip,fea_shape, False)

        images = np.stack(images)  # T,3,H,W
        activities = np.array(activities, dtype=np.int32)
        boxes = np.vstack(boxes).reshape([-1, cfg.num_boxes, 4]) # T,14,4
        boxes_real = np.vstack(boxes_real).reshape([-1, cfg.num_boxes, 4]) # T,14,4

        # convert to pytorch tensor
        images = torch.from_numpy(images).float()
        boxes = torch.from_numpy(boxes).float()
        boxes_real = torch.from_numpy(boxes_real).float()
        activities = torch.from_numpy(activities).long() # T,

        # make into batch
        b_videos.append(images)
        b_boxes.append(boxes)
        b_boxes_real.append(boxes_real)
        b_activities.append(activities)

    # batching videos
    videos = torch.stack(b_videos, dim=0).to(device)
    activities_in = torch.stack(b_activities, dim=0).to(device)
    activities_in = activities_in[:, 0]
    bboxes = torch.stack(b_boxes, dim=0).to(device)
    return label_info, offsets, videos, activities_in, bboxes, b_boxes_real

if __name__ == '__main__':
    import sys
    fig = sys.argv[1]
    cfg = get_config(fig)
    device = torch.device(0)
    cfg.init_config(rank=device)

    np.random.seed(cfg.train_random_seed)
    torch.manual_seed(cfg.train_random_seed)
    random.seed(cfg.train_random_seed)

    # Build model and optimizer
    basenet_list = {'NBA': Basenet_NBA}
    gcnnet_list = {
    }

    if cfg.training_stage == 1:
        Basenet = basenet_list[cfg.dataset_name]
        model = Basenet(cfg)
        print("load model")
        model.loadmodel(cfg.stage1_model_path)
        
    elif cfg.training_stage == 2:
        GCNnet = gcnnet_list[cfg.inference_module_name]
        model = GCNnet(cfg)
        # Load backbone
        if cfg.load_backbone_stage2:
            # 载入一阶段的模型
            model.loadmodel(cfg.stage1_model_path)
        elif cfg.load_stage2model:
            state = torch.load(cfg.stage2model)
            model.load_state_dict(state['state_dict'])
            print_log(cfg.log_path, 'Loading stage2 model: ' + cfg.stage2model)
        else:
            print_log(cfg.log_path, 'Not loading stage1 or stage2 model.')
   
    model = model.to(device)
    model.eval()
    videos_dir = 'plot_demo/NBA'

    labels, offsets, videos, activities_in, bboxes, b_boxes_real = get_batch_data(cfg,videos_dir,device)

    result_path = 'result/plot_inference'
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    # batch inference
    with torch.no_grad():
        print("videos:", videos[0]==videos[1])

        ret = model(videos, bboxes)
        print("ret:",ret['activities'])
        print("softmax ret:",torch.softmax(ret['activities'], dim=1))
        if 'activities' in list(ret.keys()):
            activities_pred, activities_labels = torch.max(torch.softmax(ret['activities'], dim=1), dim=1)
            # activities_labels = torch.argmax(activities_scores, dim=1)
            activities_correct = torch.sum(torch.eq(activities_labels.int(), activities_in.int()).float())
            activities_accuracy = activities_correct.item() / activities_labels.shape[0]

            act_pred = activities_pred.cpu().numpy()
            act_labels = activities_labels.cpu().numpy()
            act_in = activities_in.cpu().numpy()
            print("pre:",act_labels)
            print("lab:",act_in)
            print("prob:",act_pred)

            video_nums = len(act_in)
            for i in range(video_nums):
                video_id = labels[i][0]
                video = videos[i]
                bbox = b_boxes_real[i]
                result_vid = os.path.join(result_path,str(video_id))
                if not os.path.exists(result_vid):
                    os.mkdir(result_vid)

                for j,offset in enumerate(offsets):
                    img = Image.open(videos_dir + '/%d/%06d.jpg' % (video_id, offset))#.resize((224,224))
                    plot_img = draw_objs_NBA(
                            img,
                            act_labels[i],
                            act_pred[i],
                            bbox[j],
                            category_index=id_to_gact,
                            font_size=20)
                    plot_img.save(result_vid + '/' + str(offset) + '.jpg')

                

                


    
