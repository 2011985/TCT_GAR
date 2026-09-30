from datetime import datetime
import os
import platform
import random
import time
# from torchinfo import summary

import torch
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from dataload.dataset import return_dataloader
from model.base_model import Basenet_volleyball, Basenet_collective, Basenet_NBA
from model.gcn_model import GCNnet_volleyball, GCNnet_collective
from model.trans_model3 import Trans_volleyball
from model.infer_model import Dynamic_volleyball, Dynamic_TCE_volleyball, PCTDM_volleyball, HiGCIN_volleyball, \
    AT_volleyball, \
    ARG_volleyball, SACRF_BiUTE_volleyball, Dynamic_collective, \
    SAM_NBA
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from utils import collect_results_for_analysis, getLogger, print_log, show_epoch_info, show_config, getModelSize, AverageMeter, Timer, ConfusionMeter, MPCA, \
    plot_classes_preds
from thop import profile


# learn more about batch norm!
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


# update lr to new_lr
def adjust_lr(optimizer, new_lr):
    # print('change learning rate:', new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def count_flops(model, input_shape):
    # 假设模型的输入形状为 (batch_size, channels, height, width)
    # 根据实际情况调整input_shape
    input_data0 = torch.randn(input_shape[0])
    input_data1 = torch.randn(input_shape[1])
    input_data2 = torch.randn(input_shape[2])
    macs, params = profile(model, inputs=(input_data0,input_data1,input_data2), verbose=False)
    return macs  # 返回FLOPs（乘积累加操作，MACs）


def train_net(cfg):
    """
    training gcn net
    """

    rank = dist.get_rank() # 记录线程号

    # Show config parameters
    cfg.init_config(rank)
    log_path = os.path.join(cfg.result_path, 'runs')
    tb_logger = SummaryWriter(log_path) # tensorboard记录结果
    if cfg.dataset_name == 'volleyball':
        cfg.classes = ['r_set', 'r_spike', 'r-pass', 'r_winpoint',
                       'l_set', 'l-spike', 'l-pass', 'l_winpoint']

    curr_time = datetime.now().strftime("%Y-%m-%dT%H-%M-%SZ")

    logfile_path = os.path.join('./result/%s' % cfg.exp_name, curr_time + '.log')

    logger = getLogger(name=__name__, path=logfile_path)
    # for k, v in cfg.__dict__.items():
    #     logger.info(f"{k}: {v}\n")

    logger.info("Working config: {}\n".format(cfg.__dict__.items()))
    logger.info("Host: {}".format(platform.node()))
    logger.info("Logfile will be saved in {}".format(logfile_path))

    # show_config(cfg)
    # Set random seed
    np.random.seed(cfg.train_random_seed)
    torch.manual_seed(cfg.train_random_seed)
    random.seed(cfg.train_random_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # build dataloader
    training_loader, validation_loader = return_dataloader(cfg)
    logger.info('total number of clips is {} for training data'.format(training_loader.dataset.__len__()))
    logger.info('total number of clips is {} for testing data'.format(validation_loader.dataset.__len__()))
    logger.info('using position encoding: {}'.format(cfg.spatial_encoding))

    # Build model and optimizer
    basenet_list = {'volleyball': Trans_volleyball, 'collective': Basenet_collective, 'NBA': Basenet_NBA}
    gcnnet_list = {
        'SAM_NBA': SAM_NBA,
        'volleyball': GCNnet_volleyball,
        'collective': GCNnet_collective,
        'dynamic_volleyball': Dynamic_volleyball,
        'dynamic_tce_volleyball': Dynamic_TCE_volleyball,
        'pctdm_volleyball': PCTDM_volleyball,
        'higcin_volleyball': HiGCIN_volleyball,
        'at_volleyball': AT_volleyball,
        'arg_volleyball': ARG_volleyball,
        'sacrf_biute_volleyball': SACRF_BiUTE_volleyball,
        'dynamic_collective': Dynamic_collective,
        'trans_volleyball': Trans_volleyball
    }

    if cfg.training_stage == 1:
        Basenet = basenet_list[cfg.dataset_name]
        model = Basenet(cfg)
    elif cfg.training_stage == 2:
        GCNnet = gcnnet_list[cfg.inference_module_name]
        model = GCNnet(cfg)
        # Load backbone
        if cfg.load_backbone_stage2:
            # 载入一阶段的模型
            model.loadmodel(cfg.stage1_model_path)
            if cfg.load_posemodel:
                model.loadpose(cfg.pose_model_path)
        elif cfg.load_stage2model:
            state = torch.load(cfg.stage2model)
            model.load_state_dict(state['state_dict'])
            # print_log(cfg.log_path, 'Loading stage2 model: ' + cfg.stage2model)
        else:
            print_log(cfg.log_path, 'Not loading stage1 or stage2 model.')
    else:
        assert False

    # for batch_data in validation_loader:
    #     if rank == 0:
    #         tb_logger.add_graph(model, [batch_data[0], batch_data[1]])
    #     break
    
    # input_shape_list = [(2, 10, 3, 720, 1280),(2,10,12,4),(2,10,12,17,3)]
    # flops = count_flops(model, input_shape_list)
    # logger.info('FLOPs: {} million'.format(flops))
    # print(f"FLOPs: {flops} million")
    model = model.to(rank)
    # model.savepose('result/pose_trans_84.22%.pth')
    # if rank == 0:
    #     logger.info('Generate network structure')
    #     input0 = torch.ones((2, 1, 3, 720, 1280), device=rank)
    #     input1 = torch.ones((2, 1, 12, 4), device=rank)
    #     input2 = torch.ones((2,12,17,3), device=rank)
    #     # 可视化网络结构
    #     tb_logger.add_graph(model, [input0, input1,input2])
    # input0 = torch.ones((2, 1, 3, 720, 1280), device=rank)
    # input1 = torch.ones((2, 1, 12, 4), device=rank)
    # input2 = torch.ones((2,12,17,3), device=rank)
    # model = DDP(model, device_ids=[rank],find_unused_parameters=True)
    # summary(model,[(4, 1, 3, 720, 1280),(4, 1, 12, 4), (4,12,17,3)])
    # with SummaryWriter(comment='myNet') as w:
    #     w.add_graph(model, [input0,input1, input2]) 
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params: {}'.format(n_parameters))
    

    model.train()
    # 是否冻结BN层
    if cfg.set_bn_eval:
        model.apply(set_bn_eval)

    # 查看模型大小
    getModelSize(model)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.train_learning_rate,
                           weight_decay=cfg.weight_decay)

    train_list = {'volleyball': train_volleyball, 'collective': train_collective,'NBA':train_volleyball}
    test_list = {'volleyball': test_volleyball, 'collective': test_collective,'NBA':test_volleyball}
    train = train_list[cfg.dataset_name]
    test = test_list[cfg.dataset_name]

    if cfg.test_before_train:
        test_info = test(validation_loader, model, rank, 0, cfg)
        # print(test_info)

    # Training iteration
    # put the best result to the dict
    best_result = {'epoch': 0, 'activities_acc': 0}
    start_epoch = 1
    best_epoch = 1
    # print(f'model modules: {[m for m in model.modules()]}')
    for epoch in range(start_epoch, start_epoch + cfg.max_epoch):
        training_loader.sampler.set_epoch(epoch)
        # change lr
        if epoch in cfg.lr_plan:
            adjust_lr(optimizer, cfg.lr_plan[epoch])

        # One epoch of forward and backward
        train_info = train(logger, training_loader, model, rank, optimizer, epoch, cfg)
        logger.info(f"Finished Training epoch-{epoch}")

        # if rank == 0:
        #     show_epoch_info('Train', cfg.log_path, train_info)

        # Test
        if epoch % cfg.test_interval_epoch == 0:
            test_info,results = test(logger, validation_loader, model, rank, epoch, cfg)
            logger.info(f"Finished Testing epoch-{epoch}")
            if rank == 0:
                # show_epoch_info('Test', cfg.log_path, test_info)

                # write to tensorboard
                tb_logger.add_scalar("train_loss", train_info['loss'], epoch)
                tb_logger.add_scalar("activities_accuracy", test_info['activities_acc'], epoch)
                tb_logger.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

                if test_info['activities_acc'] > best_result['activities_acc']:
                    best_result = test_info
                    best_epoch = epoch
                    collect_results_for_analysis(results, validation_loader.dataset.class2idx, dataset_name=cfg.dataset_name,
                    result_prefix=logfile_path.split('.log')[0] + '_result', person_action_predicted=True, logger=logger)
                # print_log(cfg.log_path,
                #           'Best group activity accuracy: %.2f%% at epoch #%d.' % (
                #               best_result['activities_acc'], best_result['epoch']))
                if cfg.add_figure:
                    fig = plot_classes_preds(cfg,
                                             net=model,
                                             videos_dir="./plot_demo/volleyball",
                                             num_plot=5,
                                             rank=rank)
                    if fig is not None:
                        tb_logger.add_figure("predictions vs. actuals",
                                             figure=fig,
                                             global_step=epoch)
                # tb_logger.add_histogram("conv1",
                #                         values=model.conv1.weight,
                #                         global_step=epoch)

            # Save model or not
            if cfg.training_stage == 2:
                if rank == 0:
                    state = {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    filepath = cfg.result_path + '/stage%d_epoch%d_%.2f%%.pth' % (
                        cfg.training_stage, epoch, test_info['activities_acc'])
                    if cfg.load_posemodel == True:
                        torch.save(state, filepath)
                    else:
                        torch.save(state, filepath)
                        # model.savepose(filepath)
                    # print('model saved to:', filepath)
            elif cfg.training_stage == 1:
                if rank == 0:
                    if test_info['activities_acc'] == best_result['activities_acc']:
                        for m in model.modules():
                            if isinstance(m, Basenet):
                                filepath = cfg.result_path + '/stage%d_epoch%d_%.2f%%.pth' % (
                                    cfg.training_stage, epoch, test_info['activities_acc'])
                                m.savemodel(filepath)
                                # print('model saved to:',filepath)
            else:
                assert False

            logger.info(f"Test Prec@1: {test_info['activities_acc']:.3f} % / top@1: {test_info['top1']:.3f} % / Person Prec@1: {test_info['actions_acc']:.3f} % / person Top1@1: {test_info['person_top1']:.3f} % @epoch-{epoch}")
            logger.info(f"Best Test Prec@1: {best_result['activities_acc']:.3f} % / top@1: {best_result['top1']:.3f} % / Person Prec@1: {best_result['actions_acc']:.3f} % / person Top1@1: {best_result['person_top1']:.3f} % @epoch-{best_epoch}")



def train_volleyball(logger, data_loader, model, device, optimizer, epoch, cfg):
    # define metrics
    actions_meter = AverageMeter()
    activities_meter = AverageMeter()
    loss_meter = AverageMeter()
    epoch_timer = Timer()
    model.train()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    activities_conf = ConfusionMeter(cfg.num_activities)
    batch_start_time = time.time()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    top1_person = AverageMeter()
    top3_person = AverageMeter()

    for i,batch_data in enumerate(data_loader):
        if cfg.set_bn_eval:
            model.apply(set_bn_eval)

        # prepare batch data
        # (b,t,...)
        data_time.update(time.time() - batch_start_time)

        # move to GPU
        batch_data = [b.to(device=device) for b in batch_data]
        batch_size = batch_data[0].shape[0]
        num_frames = batch_data[0].shape[1]

        actions_in = batch_data[2]
        activities_in = batch_data[3]

        gt_boxes = batch_data[1]
        # seq = np.argsort((gt_boxes[:,:,:,0] + gt_boxes[:,:,:,2]).cpu().detach().numpy(),axis=2)[0][0]
        actions_in = actions_in[:, 0].reshape((batch_size * cfg.num_boxes))
        activities_in = activities_in[:, 0].reshape((batch_size,))
        

        # forward
        ret = model(batch_data[0], batch_data[1],batch_data[4])
        # action_scores shape:torch.Size([48, 9])
        # activities scores shape:torch.Size([4, 8])

        loss_list = []

        # Predict activities
        if 'activities' in list(ret.keys()):
            activities_scores = ret['activities']

            if isinstance(activities_scores, list):
                activities_loss = sum(
                [F.cross_entropy(activity, activities_in) for activity in activities_scores]) / world_size
            else:
                activities_loss = F.cross_entropy(activities_scores, activities_in) / world_size
            
            if isinstance(activities_scores, list):
                activities_scores = sum(activities_scores)
            
            activities_labels = torch.argmax(activities_scores, dim=1)
            activities_correct = torch.sum(torch.eq(activities_labels.int(), activities_in.int()).float())
            dist.barrier()
            dist.all_reduce(activities_loss)
            dist.all_reduce(activities_correct)
            activities_accuracy = activities_correct.item() / (activities_scores.shape[0] * world_size)
            if rank == 0:
                loss_list.append(activities_loss)
                activities_meter.update(activities_accuracy, activities_scores.shape[0] * world_size)
                activities_conf.add(activities_labels, activities_in)

        # Predict actions
        if 'actions' in list(ret.keys()):
            actions_scores = ret['actions']
            actions_weights = torch.tensor(cfg.actions_weights).to(device=device)

            if isinstance(actions_scores, list):
                actions_loss = sum(
                [F.cross_entropy(action, actions_in, weight=actions_weights) for action in actions_scores]) / world_size
            else:
                actions_loss = F.cross_entropy(actions_scores, actions_in, weight=actions_weights) / world_size
            if isinstance(actions_scores, list):
                actions_scores = sum(actions_scores)
            actions_labels = torch.argmax(actions_scores, dim=1)
            actions_correct = torch.sum(torch.eq(actions_labels.int(), actions_in.int()).float())
            dist.barrier()
            dist.all_reduce(actions_correct)
            dist.all_reduce(actions_loss)
            actions_accuracy = actions_correct.item() / (actions_scores.shape[0] * world_size)
            if rank == 0:
                loss_list.append(actions_loss)
                actions_meter.update(actions_accuracy, actions_scores.shape[0] * world_size)

        if 'halting' in list(ret.keys()):
            if rank == 0:
                loss_list.append(ret['halting'] * cfg.halting_penalty)

        # Update Total loss
        total_loss = sum(loss_list)
        dist.barrier()
        dist.all_reduce(actions_loss)
        # aux_loss = ret['aux']

        # if rank==0:
        #     print(loss_list)
        #     print(total_loss)
        # exit()
        if rank == 0:
            loss_meter.update(total_loss.item(), batch_size)
            top1.update(activities_accuracy, activities_scores.shape[0] * world_size)
            # top3.update(prec3.item(), activities_scores.shape[0] * world_size)
            top1_person.update(actions_accuracy, actions_scores.shape[0] * world_size)
            # top3_person.update(prec3_person.item(), actions_scores.shape[0] * world_size)
        # Optim
        optimizer.zero_grad()
        total_loss.backward()
        # total_loss.backward(retain_graph=True)
        # aux_loss.backward()
        optimizer.step()

        # finish
        batch_time.update(time.time() - batch_start_time)
        batch_start_time = time.time()
        logger.info('Train [e{0:02d}][{1}/{2}] '
                  'Time {batch_time.val:.3f}({batch_time.avg:.3f}) '
                  'Data {data_time.val:.3f}({data_time.avg:.3f}) '
                  'Loss {loss.val:.4f}({loss.avg:.4f}) '
                #   'Loss Constrastive {constras_loss.val:.4f}({constras_loss.avg:.4f}) '
                  'Top1 {top1.val:.4f}({top1.avg:.4f}) '
                #   'Top3 {top3.val:.4f}({top3.avg:.4f}) '
                  'Person Top1 {top1_person.val:.4f}({top1_person.avg:.4f}) '.format(
                #   'Person Top3 {top3_person.val:.4f}({top3_person.avg:.4f}) '.format(
                      epoch, i+1, len(data_loader), batch_time=batch_time,
                      data_time=data_time,
                      loss=loss_meter, top1=top1, top3=top3,
                      top1_person=top1_person))

    train_info = {
        'time': epoch_timer.timeit(),
        'epoch': epoch,
        'loss': loss_meter.avg,
        'top1':top1.avg*100,
        'person_top1':top1_person.avg *100,
        'activities_acc': activities_meter.avg * 100,
        'actions_acc': actions_meter.avg * 100,
        'activities_conf': activities_conf.value(),
        'activities_MPCA': MPCA(activities_conf.value())
    }

    return train_info


def test_volleyball(logger,data_loader, model, device, epoch, cfg):
    model.eval()

    actions_meter = AverageMeter()
    activities_meter = AverageMeter()
    loss_meter = AverageMeter()
    activities_conf = ConfusionMeter(cfg.num_activities)
    epoch_timer = Timer()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    batch_start_time = time.time()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    top1_person = AverageMeter()
    top3_person = AverageMeter()
    results = dict()
    with torch.no_grad():
        for i, batch_data_test in enumerate(data_loader):
            data_time.update(time.time() - batch_start_time)

            # prepare batch data
            batch_data_test = [b.to(device=device) for b in batch_data_test]
            batch_size = batch_data_test[0].shape[0]
            num_frames = batch_data_test[0].shape[1]
            actions_in = batch_data_test[2]
            activities_in = batch_data_test[3]
            batch_video = batch_data_test[5];
            batch_clip = batch_data_test[6];
            # 第一个帧的所有batch的action id 向量
            gt_boxes = batch_data_test[1]
            # seq = np.argsort((gt_boxes[:,:,:,0] + gt_boxes[:,:,:,2]).cpu().detach().numpy(),axis=2)[0][0]
            actions_in = actions_in[:, 0].reshape((batch_size * cfg.num_boxes))
            activities_in = activities_in[:, 0].reshape((batch_size,))

            # forward
            # 得到个体分数和群体行为分数
            ret = model(batch_data_test[0], batch_data_test[1], batch_data_test[4])
            # action_scores shape:torch.Size([48, 9])
            # activities scores shape:torch.Size([4, 8])

            loss_list = []

            # Predict activities
            if 'activities' in list(ret.keys()):
                activities_scores = ret['activities']
                if isinstance(activities_scores, list):
                    activities_loss = sum(
                        [F.cross_entropy(activity, activities_in) for activity in activities_scores]) / world_size
                else:
                    activities_loss = F.cross_entropy(activities_scores, activities_in) / world_size

                if isinstance(activities_scores, list):
                    activities_scores = sum(activities_scores)

                activities_labels = torch.argmax(activities_scores, dim=1)
                activities_correct = torch.sum(torch.eq(activities_labels.int(), activities_in.int()).float())
                dist.barrier()
                dist.all_reduce(activities_loss)
                dist.all_reduce(activities_correct)
                activities_accuracy = activities_correct.item() / (activities_scores.shape[0] * world_size)
                if rank == 0:
                    loss_list.append(activities_loss)
                    activities_meter.update(activities_accuracy, activities_scores.shape[0] * world_size)
                    activities_conf.add(activities_labels, activities_in)

            # Predict actions
            if 'actions' in list(ret.keys()):
                actions_scores = ret['actions']
                actions_weights = torch.tensor(cfg.actions_weights).to(device=device)

                if isinstance(actions_scores, list):
                    actions_loss = sum(
                    [F.cross_entropy(action, actions_in, weight=actions_weights) for action in actions_scores]) / world_size
                else:
                    actions_loss = F.cross_entropy(actions_scores, actions_in, weight=actions_weights) / world_size
                if isinstance(actions_scores, list):
                    actions_scores = sum(actions_scores)
              
                actions_labels = torch.argmax(actions_scores, dim=1)
                actions_correct = torch.sum(torch.eq(actions_labels.int(), actions_in.int()).float())
                dist.barrier()
                dist.all_reduce(actions_correct)
                dist.all_reduce(actions_loss)
                actions_accuracy = actions_correct.item() / (actions_scores.shape[0] * world_size)
                if rank == 0:
                    loss_list.append(actions_loss)
                    actions_meter.update(actions_accuracy, actions_scores.shape[0] * world_size)

            if 'halting' in list(ret.keys()):
                if rank == 0:
                    loss_list.append(ret['halting'] * cfg.halting_penalty)

            # Update Total loss
            total_loss = sum(loss_list)
            if rank == 0:
                loss_meter.update(total_loss.item(), batch_size)
                top1.update(activities_accuracy, activities_scores.shape[0] * world_size)
                # top3.update(prec3.item(), activities_scores.shape[0] * world_size)
                top1_person.update(actions_accuracy, actions_scores.shape[0] * world_size)
                # top3_person.update(prec3_person.item(), actions_scores.shape[0] * world_size)

            for b_idx in range(batch_size):
                key = 'video_{}-clip_{}'.format(batch_video[b_idx], batch_clip[b_idx])
                assert key not in results
                results[key] = dict()
                results[key]['Pred'] = activities_labels[b_idx].cpu().numpy()[()]
                results[key]['GT'] = activities_in[b_idx].cpu().numpy()[()]
                results[key]['Pred_Person'] = []
                results[key]['GT_Person'] = []
                for p_idx in range(cfg.num_boxes):
                    results[key]['Pred_Person'].append(actions_labels[b_idx*cfg.num_boxes+p_idx].cpu().numpy()[()])
                    results[key]['GT_Person'].append(actions_in[b_idx*cfg.num_boxes+p_idx].cpu().numpy()[()])

            # finish
            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()
            logger.info('Test [e{0:02d}][{1}/{2}] '
                      'Time {batch_time.val:.3f}({batch_time.avg:.3f}) '
                      'Data {data_time.val:.3f}({data_time.avg:.3f}) '
                      'Loss {loss.val:.4f}({loss.avg:.4f}) '
                      'Top1 {top1.val:.4f}({top1.avg:.4f}) '
                    #   'Top3 {top3.val:.4f}({top3.avg:.4f}) '
                      'Person Top1 {top1_person.val:.4f}({top1_person.avg:.4f}) '.format(
                    #   'Person Top3 {top3_person.val:.4f}({top3_person.avg:.4f}) '.format(
                          epoch, i+1, len(data_loader), batch_time=batch_time,
                          data_time=data_time,
                          loss=loss_meter, top1=top1,
                          top1_person=top1_person))

    test_info = {
        'time': epoch_timer.timeit(),
        'epoch': epoch,
        'loss': loss_meter.avg,
        'top1':top1.avg*100,
        'person_top1':top1_person.avg*100,
        'activities_acc': activities_meter.avg * 100,
        'actions_acc': actions_meter.avg * 100,
        'activities_conf': activities_conf.value(),
        'activities_MPCA': MPCA(activities_conf.value())
    }
    return test_info, results


def train_collective(data_loader, model, device, optimizer, epoch, cfg):
    actions_meter = AverageMeter()
    activities_meter = AverageMeter()
    loss_meter = AverageMeter()
    epoch_timer = Timer()
    for batch_data in data_loader:
        model.train()
        model.apply(set_bn_eval)

        # prepare batch data
        batch_data = [b.to(device=device) for b in batch_data]
        batch_size = batch_data[0].shape[0]
        num_frames = batch_data[0].shape[1]

        # forward
        actions_scores, activities_scores = model((batch_data[0], batch_data[1], batch_data[4]))

        actions_in = batch_data[2].reshape((batch_size, num_frames, cfg.num_boxes))
        activities_in = batch_data[3].reshape((batch_size, num_frames))
        bboxes_num = batch_data[4].reshape(batch_size, num_frames)

        actions_in_nopad = []
        if cfg.training_stage == 1:
            actions_in = actions_in.reshape((batch_size * num_frames, cfg.num_boxes,))
            bboxes_num = bboxes_num.reshape(batch_size * num_frames, )
            for bt in range(batch_size * num_frames):
                N = bboxes_num[bt]
                actions_in_nopad.append(actions_in[bt, :N])
        else:
            for b in range(batch_size):
                N = bboxes_num[b][0]
                actions_in_nopad.append(actions_in[b][0][:N])
        actions_in = torch.cat(actions_in_nopad, dim=0).reshape(-1, )  # ALL_N,

        if cfg.training_stage == 1:
            activities_in = activities_in.reshape(-1, )
        else:
            activities_in = activities_in[:, 0].reshape(batch_size, )

        # Predict actions
        actions_loss = F.cross_entropy(actions_scores, actions_in, weight=None)
        actions_labels = torch.argmax(actions_scores, dim=1)  # B*T*N,
        actions_correct = torch.sum(torch.eq(actions_labels.int(), actions_in.int()).float())

        # Predict activities
        activities_loss = F.cross_entropy(activities_scores, activities_in)
        activities_labels = torch.argmax(activities_scores, dim=1)  # B*T,
        activities_correct = torch.sum(torch.eq(activities_labels.int(), activities_in.int()).float())

        # Get accuracy
        actions_accuracy = actions_correct.item() / actions_scores.shape[0]
        activities_accuracy = activities_correct.item() / activities_scores.shape[0]

        actions_meter.update(actions_accuracy, actions_scores.shape[0])
        activities_meter.update(activities_accuracy, activities_scores.shape[0])

        # Total loss
        total_loss = activities_loss + cfg.actions_loss_weight * actions_loss
        loss_meter.update(total_loss.item(), batch_size)

        # Optim
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    train_info = {
        'time': epoch_timer.timeit(),
        'epoch': epoch,
        'loss': loss_meter.avg,
        'activities_acc': activities_meter.avg * 100,
        'actions_acc': actions_meter.avg * 100
    }

    return train_info


def test_collective(data_loader, model, device, epoch, cfg):
    model.eval()

    actions_meter = AverageMeter()
    activities_meter = AverageMeter()
    loss_meter = AverageMeter()

    epoch_timer = Timer()
    with torch.no_grad():
        for batch_data in data_loader:
            # prepare batch data
            batch_data = [b.to(device=device) for b in batch_data]
            batch_size = batch_data[0].shape[0]
            num_frames = batch_data[0].shape[1]

            actions_in = batch_data[2].reshape((batch_size, num_frames, cfg.num_boxes))
            activities_in = batch_data[3].reshape((batch_size, num_frames))
            bboxes_num = batch_data[4].reshape(batch_size, num_frames)

            # forward
            actions_scores, activities_scores = model((batch_data[0], batch_data[1], batch_data[4]))

            actions_in_nopad = []

            if cfg.training_stage == 1:
                actions_in = actions_in.reshape((batch_size * num_frames, cfg.num_boxes,))
                bboxes_num = bboxes_num.reshape(batch_size * num_frames, )
                for bt in range(batch_size * num_frames):
                    N = bboxes_num[bt]
                    actions_in_nopad.append(actions_in[bt, :N])
            else:
                for b in range(batch_size):
                    N = bboxes_num[b][0]
                    actions_in_nopad.append(actions_in[b][0][:N])
            actions_in = torch.cat(actions_in_nopad, dim=0).reshape(-1, )  # ALL_N,

            if cfg.training_stage == 1:
                activities_in = activities_in.reshape(-1, )
            else:
                activities_in = activities_in[:, 0].reshape(batch_size, )

            actions_loss = F.cross_entropy(actions_scores, actions_in)
            actions_labels = torch.argmax(actions_scores, dim=1)  # ALL_N,
            actions_correct = torch.sum(torch.eq(actions_labels.int(), actions_in.int()).float())

            # Predict activities
            activities_loss = F.cross_entropy(activities_scores, activities_in)
            activities_labels = torch.argmax(activities_scores, dim=1)  # B,
            activities_correct = torch.sum(torch.eq(activities_labels.int(), activities_in.int()).float())

            # Get accuracy
            actions_accuracy = actions_correct.item() / actions_scores.shape[0]
            activities_accuracy = activities_correct.item() / activities_scores.shape[0]

            actions_meter.update(actions_accuracy, actions_scores.shape[0])
            activities_meter.update(activities_accuracy, activities_scores.shape[0])

            # Total loss
            total_loss = activities_loss + cfg.actions_loss_weight * actions_loss
            loss_meter.update(total_loss.item(), batch_size)

    test_info = {
        'time': epoch_timer.timeit(),
        'epoch': epoch,
        'loss': loss_meter.avg,
        'activities_acc': activities_meter.avg * 100,
        'actions_acc': actions_meter.avg * 100
    }

    return test_info
