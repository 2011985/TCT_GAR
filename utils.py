import logging
import os.path
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import time
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from thop import profile, clever_format
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from dataload.volleyball import ACTIONS as VL_ACTIONS
from dataload.volleyball import ACTIVITIES as VL_ACTIVITIES

_FMT = "[%(asctime)s] %(levelname)s: %(message)s"
_DATEFMT = "%m/%d/%Y %H:%M:%S"

def fileHandler(path, format, datefmt, mode="w"):
    handler = logging.FileHandler(path, mode=mode)
    formatter = logging.Formatter(format, datefmt=datefmt)
    handler.setFormatter(formatter)
    return handler

def getLogger(
    name=None,
    path=None,
    level=logging.INFO,
    format=_FMT,
    datefmt=_DATEFMT,
):

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if path is not None:
        from pathlib import Path
        path = str(Path(path).resolve())
        if not any(
            map(lambda hdr: hasattr(hdr, 'baseName') 
                and hdr.baseFilename == path, logger.handlers)):
            handler = fileHandler(path, format, datefmt)
            logger.addHandler(handler)

    return logger


def build_mlp(input_dim, hidden_dims, output_dim=None, use_batchnorm=False, dropout=0):
    layers = []
    D = input_dim
    if dropout > 0:
        layers.append(nn.Dropout(p=dropout))
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(input_dim))
    if hidden_dims:
        for dim in hidden_dims:
            layers.append(nn.Linear(D, dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(dim))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.ReLU(inplace=True))
            D = dim
    if output_dim:
        layers.append(nn.Linear(D, output_dim))
    return nn.Sequential(*layers)

def plot_confusion_matrix(cm, classes, plot_name,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, logger=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        if logger:
            logger.info("Normalized confusion matrix")
        else:
            print("Normalized confusion matrix")
    else:
        if logger:
            logger.info('Confusion matrix, without normalization')
        else:
            print('Confusion matrix, without normalization')

    if logger:
        logger.info(cm)
    else:
        print(cm)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(plot_name, bbox_inches='tight', transparent=True, pad_inches=0)
    plt.close()
    
    if logger:
        logger.info('{} saved!'.format(plot_name))
    else:
        print('{} saved!'.format(plot_name))    
    return

def write_list_of_strings_into_a_txtfile(lines, savepath):
    textfile = open(savepath, "w")
    for line in lines:
        textfile.write(line + "\n")
    textfile.close()
    return

def collect_results_for_analysis(results, class2idx, 
                                 dataset_name='volleyball', 
                                 result_prefix='./result',
                                 person_action_predicted=True,
                                 logger=None):
    idx2class = dict()
    if dataset_name == 'volleyball':
        for c in class2idx:
            if '_' in c:
                idx2class[class2idx[c]] = c
                if logger:
                    logger.info('{}: {}'.format(class2idx[c], c))
                  
        if person_action_predicted:
            person_idx2class = {0: 'blocking', 
               1: 'digging', 
               2: 'falling', 
               3: 'jumping',
               4: 'moving', 
               5: 'setting', 
               6: 'spiking', 
               7: 'standing',
               8: 'waiting'}

        else:
            person_idx2class = None
    else:
        for c in class2idx:
            idx2class[class2idx[c]] = c
            if logger:
                logger.info('{}: {}'.format(class2idx[c], c))
                
        person_idx2class = None
     
    if logger:
        logger.info('collecting results to files...')
        
    gt_all = []
    pred_all = []
    if person_idx2class:
        gt_person_all = []
        pred_person_all = []
    lines = []
    for key in tqdm(results):
        [videoid, clipid] = key.split('-')
        gt = idx2class[results[key]['GT']]
        pred = idx2class[results[key]['Pred']]
        gt_all.append(gt)
        pred_all.append(pred)
        
        if person_idx2class:
            results[key]['GT_Person'] = [person_idx2class[val] for val in results[key]['GT_Person']]
            results[key]['Pred_Person'] = [person_idx2class[val] for val in results[key]['Pred_Person']]
            gt_person_all += results[key]['GT_Person']
            pred_person_all += results[key]['Pred_Person']
            
        lines.append("{}  GT {} Pred {} Correct? {}".format(key, gt, pred, gt==pred))
    
    # # write results to disk
    pickle.dump(results, open(result_prefix+'.pkl', "wb"))
    
    # write results to txt file
    write_list_of_strings_into_a_txtfile(lines, result_prefix+'.txt')
    
    # get confusion matrix
    cnf_matrix = confusion_matrix(gt_all, pred_all, labels=list(idx2class.values()))
    np.set_printoptions(precision=2)

    # plot normalized confusion matrix
    plot_name = result_prefix + '.png'
    plot_confusion_matrix(cnf_matrix, list(idx2class.values()), plot_name, normalize=True,
                          title='Confusion matrix, with normalization', cmap=plt.cm.Reds, logger=logger)
    
    if person_idx2class:
        # plot normalized confusion matrix for person action prediction
        cnf_matrix_person = confusion_matrix(gt_person_all, pred_person_all, labels=list(person_idx2class.values()))
        np.set_printoptions(precision=2)

        # plot normalized confusion matrix for person action prediction
        plot_name = result_prefix + '_person.png'
        plot_confusion_matrix(cnf_matrix_person, list(person_idx2class.values()), plot_name, normalize=True,
                              title='Confusion matrix, with normalization', cmap=plt.cm.Reds, logger=logger)   

    return 

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('model size: {:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)


def prep_images(images):
    """
    preprocess images
    Args:
        images: pytorch tensor
    """
    images = images.div(255.0)

    images = torch.sub(images, 0.5)
    images = torch.mul(images, 2.0)

    return images


# def prep_images(images):
#     """
#     preprocess images
#     Args:
#         images: pytorch tensor
#     """
#     # Reference: pytorch.org/docs/stable/torchvision/models.html
#
#     images = images.div(255.0)
#     normalizer = transforms.Normalize(
#         mean = [0.485, 0.456, 0.406],
#         std = [0.229, 0.224, 0.225]
#     )
#
#     for i in range(images.shape[0]):
#         images[i] = normalizer(images[i])
#     #images = torch.sub(images,0.5)
#     #images = torch.mul(images,2.0)
#
#     return images

def calc_pairwise_distance(X, Y):
    """
    computes pairwise distance between each element
    Args: 
        X: [N,D]
        Y: [M,D]
    Returns:
        dist: [N,M] matrix of euclidean distances
    """
    rx = X.pow(2).sum(dim=1).reshape((-1, 1))
    ry = Y.pow(2).sum(dim=1).reshape((-1, 1))
    dist = rx - 2.0 * X.matmul(Y.t()) + ry.t()
    return torch.sqrt(dist)


def calc_pairwise_distance_3d(X, Y):
    """
    computes pairwise distance between each element
    Args: 
        X: [B,N,D]
        Y: [B,M,D]
    Returns:
        dist: [B,N,M] matrix of euclidean distances
    """
    B = X.shape[0]

    rx = X.pow(2).sum(dim=2).reshape((B, -1, 1))
    ry = Y.pow(2).sum(dim=2).reshape((B, -1, 1))

    dist = rx - 2.0 * X.matmul(Y.transpose(1, 2)) + ry.transpose(1, 2)

    return torch.sqrt(dist)


def sincos_encoding_2d(positions, d_emb):
    """
    Args:
        positions: [N,2]
    Returns:
        positions high-dimensional representation: [N,d_emb]
    """

    N = positions.shape[0]

    d = d_emb // 2

    idxs = [np.power(1000, 2 * (idx // 2) / d) for idx in range(d)]
    idxs = torch.FloatTensor(idxs).to(device=positions.device)

    idxs = idxs.repeat(N, 2)  # N, d_emb

    pos = torch.cat([positions[:, 0].reshape(-1, 1).repeat(1, d), positions[:, 1].reshape(-1, 1).repeat(1, d)], dim=1)

    embeddings = pos / idxs

    embeddings[:, 0::2] = torch.sin(embeddings[:, 0::2])  # dim 2i
    embeddings[:, 1::2] = torch.cos(embeddings[:, 1::2])  # dim 2i+1

    return embeddings


def print_log(file_path, *args):
    print(*args)
    if file_path is not None:
        with open(file_path, 'a') as f:
            print(*args, file=f)


def show_config(cfg):
    print_log(cfg.log_path, '=====================Config=====================')
    for k, v in cfg.__dict__.items():
        print_log(cfg.log_path, k, ': ', v)
    print_log(cfg.log_path, '======================End=======================')


def show_epoch_info(phase, log_path, info):
    # print an empty line
    print_log(log_path, '')
    if phase == 'Test':
        print_log(log_path, '====> %s at epoch #%d' % (phase, info['epoch']))
    else:
        print_log(log_path, '%s at epoch #%d' % (phase, info['epoch']))

    print_log(log_path, 'Group Activity Accuracy: %.2f%%, Loss: %.5f, Using %.1f seconds' % (
        info['activities_acc'], info['loss'], info['time']))

    if 'activities_conf' in info.keys():
        print_log(log_path, info['activities_conf'])
    if 'activities_MPCA' in info.keys():
        print_log(log_path, 'Activities MPCA:{:.2f}%'.format(info['activities_MPCA']))
    if 'MAD' in info.keys():
        print_log(log_path, 'MAD:{:.4f}'.format(info['MAD']))
    print_log(log_path, '\n')


def log_final_exp_result(log_path, data_path, exp_result):
    no_display_cfg = ['num_workers', 'use_gpu', 'use_multi_gpu', 'device_list',
                      'batch_size_test', 'test_interval_epoch', 'train_random_seed',
                      'result_path', 'log_path', 'device']

    with open(log_path, 'a') as f:
        print('', file=f)
        print('', file=f)
        print('', file=f)
        print('=====================Config=====================', file=f)

        for k, v in exp_result['cfg'].__dict__.items():
            if k not in no_display_cfg:
                print(k, ': ', v, file=f)

        print('=====================Result======================', file=f)

        print('Best result:', file=f)
        print(exp_result['best_result'], file=f)

        print('Cost total %.4f hours.' % (exp_result['total_time']), file=f)

        print('======================End=======================', file=f)

    data_dict = pickle.load(open(data_path, 'rb'))
    data_dict[exp_result['cfg'].exp_name] = exp_result
    pickle.dump(data_dict, open(data_path, 'wb'))


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def plot_classes_preds(cfg, net, videos_dir, num_plot=5, rank=0):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    if not os.path.exists(videos_dir):
        print(" not found {} path".format(videos_dir))
        return None
    label_path = os.path.join(videos_dir, "label.txt")
    if not os.path.exists(label_path):
        print(" not found {} file".format(label_path))
        return None
    gact_to_id = {name: i for i, name in enumerate(VL_ACTIVITIES)}
    act_to_id = {name: i for i, name in enumerate(VL_ACTIONS)}

    label_info = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            # 每一行数据存在value里
            values = line.split(' ')
            file_name = values[0]

            # 每个标注行的文件id,实际上就是clip id, 对应的图像序列在标注文件同级下的相同id的目录下
            fid = file_name.split('.')[0]
            clip_path = os.path.join(videos_dir,fid)
            file_path = os.path.join(clip_path, file_name)
            if not os.path.exists(clip_path):
                print("not found {}, skip".format(clip_path))
                continue
            if values[1] not in gact_to_id.keys():
                print("unrecognized category {}, skip".format(values[1]))
                continue
            activity = gact_to_id[values[1]]

            # 个体bounding box以及类别标注
            values = values[2:]
            num_people = len(values) // 5
            # store all the actions in the scene
            action_names = values[4::5]
            actions = [act_to_id[name]
                       for name in action_names]

            def _read_bbox(xywh):
                x, y, w, h = map(int, xywh)
                return y, x, y + h, x + w

            # 每个人的边界框存在bboxes中 , bbox's shape (12,4)(11,4)(10,4)均可能
            bboxes = np.array([_read_bbox(values[i:i + 4])
                               for i in range(0, 5 * num_people, 5)])

            
            label_info.append([int(fid), file_path, activity, actions, bboxes])
            # annotations[fid] = {
            #     'file_name': file_name,
            #     'group_activity': activity,
            #     'actions': actions,
            #     'bboxes': bboxes,
            # }
    if len(label_info) > num_plot:
        label_info = label_info[:num_plot]

    num_videos = len(label_info)
    all_tracks = pickle.load(open(cfg.data_path + '/tracks_normalized.pkl', 'rb'))
    OH, OW = cfg.out_size

    b_videos = []
    b_activities = []
    b_boxes = []
    b_actions = []
    for enti in label_info:
        src_fid, file_path, activity, action, bboxes = enti
        images, boxes = [], []
        activities, actions = [], []
        frames_sample = [(src_fid, fid) for fid in
                         range(src_fid - cfg.num_before, src_fid + cfg.num_after + 1)]
        for i, (src_fid, fid) in enumerate(frames_sample):
            img = Image.open(videos_dir + '/%d/%d.jpg' % (src_fid, fid))
            # 对image的大小进行转换
            img = transforms.functional.resize(img, cfg.image_size)
            img = np.array(img)

            # H,W,3 -> 3,H,W
            img = img.transpose(2, 0, 1)
            images.append(img)

            temp_boxes = np.ones_like(all_tracks[(0, src_fid)][fid])

            for i, track in enumerate(all_tracks[(0, src_fid)][fid]):
                y1, x1, y2, x2 = track
                # multiply output channel shape
                w1, h1, w2, h2 = x1 * OW, y1 * OH, x2 * OW, y2 * OH
                temp_boxes[i] = np.array([w1, h1, w2, h2])

            boxes.append(temp_boxes)

            actions.append(action)

            if len(boxes[-1]) != cfg.num_boxes:
                # if not equal , copy the missing ones to make it equal
                boxes[-1] = np.vstack([boxes[-1], boxes[-1][:cfg.num_boxes - len(boxes[-1])]])
                actions[-1] = actions[-1] + actions[-1][:cfg.num_boxes - len(actions[-1])]

            activities.append(activity)

        images = np.stack(images)  # T,3,H,W
        activities = np.array(activities, dtype=np.int32)
        bboxes = np.vstack(boxes).reshape([-1, cfg.num_boxes, 4])  # T,12,4
        actions = np.hstack(actions).reshape([-1, cfg.num_boxes])  # T,12

        # convert to pytorch tensor
        images = torch.from_numpy(images).float()
        bboxes = torch.from_numpy(bboxes).float()
        actions = torch.from_numpy(actions).long()
        activities = torch.from_numpy(activities).long()
        # make into batch
        b_videos.append(images)
        b_boxes.append(bboxes)
        b_actions.append(actions)
        b_activities.append(activities)

    # batching videos
    videos = torch.stack(b_videos, dim=0).to(rank)
    activities_in = torch.stack(b_activities, dim=0).to(rank)
    activities_in = activities_in[:, 0]
    bboxes = torch.stack(b_boxes, dim=0).to(rank)

    # inference
    with torch.no_grad():
        ret = net(videos, bboxes)
        if 'activities' in list(ret.keys()):
            activities_pred, activities_labels = torch.max(torch.softmax(ret['activities'], dim=1), dim=1)
            # activities_labels = torch.argmax(activities_scores, dim=1)
            activities_correct = torch.sum(torch.eq(activities_labels.int(), activities_in.int()).float())
            activities_accuracy = activities_correct.item() / activities_labels.shape[0]

            act_pred = activities_pred.cpu().numpy()
            act_labels = activities_labels.cpu().numpy()
            act_in = activities_in.cpu().numpy()

    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(num_videos * 3, 4), dpi=100)

    for i in range(num_videos):
        ax = fig.add_subplot(1, num_videos, i + 1, xticks=[], yticks=[])
        img = Image.open(label_info[i][1])
        img = transforms.functional.resize(img, cfg.image_size)
        img = np.array(img)
        plt.imshow(img.astype('uint8'))

        title = "{}, {:.2f}%\n(label: {})".format(
            VL_ACTIVITIES[act_labels[i]],  # pred class
            act_pred[i] * 100,  # pred probability
            VL_ACTIVITIES[act_in[i]]  # true class
        )
        ax.set_title(title,color=("green" if act_labels[i]==act_in[i] else "red"))

    return fig


class AverageMeter(object):
    """
    Computes the average value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    """
    class to do timekeeping
    """

    def __init__(self):
        self.last_time = time.time()

    def timeit(self):
        old_time = self.last_time
        self.last_time = time.time()
        return self.last_time - old_time


class ConfusionMeter(object):
    """Maintains a confusion matrix for a given calssification problem.

    The ConfusionMeter constructs a confusion matrix for a multi-class
    classification problems. It does not support multi-label, multi-class problems:
    for such problems, please use MultiLabelConfusionMeter.

    Args:
        k (int): number of classes in the classification problem
        normalized (boolean): Determines whether or not the confusion matrix
            is normalized or not

    """

    def __init__(self, k, normalized=False):
        super(ConfusionMeter, self).__init__()
        self.conf = np.ndarray((k, k), dtype=np.int32)
        self.normalized = normalized
        self.k = k
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, predicted, target):
        """Computes the confusion matrix of K x K size where K is no of classes

        Args:
            predicted (tensor): Can be an N x K tensor of predicted scores obtained from
                the model for N examples and K classes or an N-tensor of
                integer values between 0 and K-1.
            target (tensor): Can be a N-tensor of integer values assumed to be integer
                values between 0 and K-1 or N x K tensor, where targets are
                assumed to be provided as one-hot vectors

        """
        predicted = predicted.cpu().numpy()
        target = target.cpu().numpy()

        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        if np.ndim(predicted) != 1:
            assert predicted.shape[1] == self.k, \
                'number of predictions does not match size of confusion matrix'
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 1 and k'

        onehot_target = np.ndim(target) != 1
        if onehot_target:
            assert target.shape[1] == self.k, \
                'Onehot target does not match size of confusion matrix'
            assert (target >= 0).all() and (target <= 1).all(), \
                'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), \
                'multi-label setting is not supported'
            target = np.argmax(target, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 0 and k-1'

        # hack for bincounting 2 arrays together
        x = predicted + self.k * target
        bincount_2d = np.bincount(x.astype(np.int32),
                                  minlength=self.k ** 2)

        assert bincount_2d.size == self.k ** 2
        conf = bincount_2d.reshape((self.k, self.k))

        self.conf += conf

    def value(self):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf


def MPCA(conf_mat):
    '''

    :param conf_mat: np.ndarray((k, k), dtype=np.int32)
    :return:
    '''
    class_sum = np.sum(conf_mat, axis=1, dtype=np.float32)
    for i in range(len(class_sum)):
        class_sum[i] = np.float32(conf_mat[i][i]) / np.float32(class_sum[i])
    mpca = np.mean(class_sum) * 100
    return mpca


def MAC2FLOP(macs, params, module_name=''):
    macs, params = clever_format([macs, params], "%.3f")
    print('{} MACs: {}  #Params: {}'.format(module_name, macs, params))
    if 'M' in macs:
        flops = float(macs.replace('M', '')) * 2
        flops = str(flops / 1000) + 'G'
    elif 'G' in macs:
        flops = str(float(macs.replace('G', '')) * 2) + 'G'
    print('{} GFLOPs: {}  #Params: {}'.format(module_name, flops, params))


class MADmeter(object):
    def __init__(self, T, N):
        super(MADmeter, self).__init__()
        self.T = T
        self.N = N
        self.B = 0
        self.MAD = 0.

    def generate_mask(self, features, field, field_shape='rect'):
        if field_shape == 'rect':
            B, T, N, NFB = features.shape
            TN = T * N
            if len(field) == 1:  # shape like [3, 3] [5, 5] [7, 7] [9, 9]...
                assert field[0] % 2 == 1
                mask = torch.zeros((TN, TN), dtype=torch.bool, device=features.device)
                for i in range(TN):
                    x, y = i // N, i % N
                    for j in range(field[0]):
                        jx = j - field[0] // 2
                        if jx + x >= 0:
                            for k in range(field[0]):
                                ky = k - field[0] // 2
                                if ky + y >= 0:
                                    mask[i][(jx + x) * T + (ky + y)] = True
            elif len(field) == 2 and field[0] == T and field[1] == N:  # fully-connected
                mask = torch.ones((TN, TN), dtype=torch.bool, device=features.device)

        elif field_shape == 'dynamic':  # [B, TN, k2+1, NFB]
            B, TN, k2, NFB = features.shape
            mask = torch.zeros((TN * k2, TN * k2), dtype=torch.bool, device=features.device)
            for i in range(TN):
                for j in range(k2 - 1):
                    mask[i * k2, i * k2 + j + 1] = True

            # if len(field) == 2 and field[0] == T and field[1] == N:  # fully-connected
            #     mask = torch.ones((TN, TN), dtype=torch.bool, device=features.device)
            # else: # shape like [3, 3] [5, 5] [7, 7] [9, 9]...
            #     assert field[0]%2 == 1 and field[1]%2 == 1
            #     mask = torch.zeros((TN, TN), dtype = torch.bool, device = features.device)
            #     for i in range(TN):
            #         x, y = i//N, i%N
            #         for j in range(field[0]):
            #             jx = j - field[0] // 2
            #             if jx + x >=0:
            #                 for k in range(field[0]):
            #                     ky = k - field[0]//2
            #                     if ky + y >=0:
            #                         mask[i][(jx + x)*T + (ky + y)] = True

        return mask

    def node_select(self, features, field_shape='dynamic'):
        if field_shape == 'dynamic':
            B, TN, k2, NFB = features.shape
            node_selector = torch.zeros((TN * k2,), dtype=torch.bool, device=features.device)
            for i in range(TN):
                node_selector[i * k2] = True
            return node_selector

    def cal_MAD(self, features, field, field_shape='rect'):
        B, T, N, NFB = features.shape

        mask = self.generate_mask(features=features,
                                  field=field,
                                  field_shape=field_shape)
        features = features.view(B, T * N, NFB)
        norm = torch.norm(features, dim=2).unsqueeze(dim=2)
        divisor = torch.bmm(norm, norm.transpose(1, 2))
        dist_array = 1. - torch.bmm(features, features.transpose(1, 2)) / (divisor + 1e-8)
        dist_array = dist_array * mask.float()
        # print(mask)
        MAD = (torch.sum(dist_array, dim=2) / (torch.sum(mask.float(), dim=1) + 1e-8)).detach()  # B, TN
        # print(MAD)
        if field_shape == 'dynamic':
            node_selector = self.node_select(features.view(B, T, N, NFB), field_shape='dynamic')
            batch_MAD = MAD[node_selector[None, :].repeat((B, 1))].view(B, -1)
            self.MAD += torch.sum(torch.mean(batch_MAD, dim=1), dim=0)
        else:
            self.MAD += torch.sum(torch.mean(MAD, dim=1))
        self.B += B

    def output_MAD(self):
        return self.MAD / self.B


if __name__ == '__main__':
    # Test MPCA
    # x = np.ndarray((3,3), dtype = np.int32)
    # x[0] = [1, 2, 3]
    # x[1] = [1, 2, 3]
    # x[2] = [2, 2, 3]
    # print(MPCA(x))

    # Test dynamic MAD
    # Mad = MADmeter(10, 12)
    # f = torch.randn((1, 120, 26, 1024))
    # Mad.cal_MAD(f, [1, 25], field_shape = 'dynamic')
    # print(Mad.output_MAD())

    Mad = MADmeter(10, 12)
    f = torch.randn((1, 10, 12, 1024))
    Mad.cal_MAD(f, [10, 12], field_shape='rect')
    print(Mad.output_MAD())
