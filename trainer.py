class Trainer():
    def __init__(self, cfg,  model, logger, optimizer, train_loader=None, test_loader=None):
        self.cfg = cfg
        self.logger = logger
        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train(self, epoch):
        actions_meter = AverageMeter()
        activities_meter = AverageMeter()
        loss_meter = AverageMeter()
        epoch_timer = Timer()
        model.train()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        activities_conf = ConfusionMeter(self.cfg.num_activities)
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        top1 = AverageMeter()
        top3 = AverageMeter()
        top1_person = AverageMeter()
        top3_person = AverageMeter()

        for i,batch_data in enumerate(self.train_loader):
            if self.cfg.set_bn_eval:
                self.model.apply(set_bn_eval)

            # prepare batch data
            # (b,t,...)
            data_time.update(time.time() - batch_start_time)

            # 放入GPU
            batch_data = [b.to(device=device) for b in batch_data]
            batch_size = batch_data[0].shape[0]
            num_frames = batch_data[0].shape[1]

            actions_in = batch_data[2]
            activities_in = batch_data[3]

            # 第一个帧的所有batch的action id 向量
            actions_in = actions_in[:, 0, :].reshape((batch_size * cfg.num_boxes))
            activities_in = activities_in[:, 0].reshape((batch_size,))

            # forward
            # 得到个体分数和群体行为分数
            ret = model(batch_data[0], batch_data[1])
            # action_scores shape:torch.Size([48, 9])
            # activities scores shape:torch.Size([4, 8])

            loss_list = []

            # Predict activities
            if 'activities' in list(ret.keys()):
                activities_scores = ret['activities']
                activities_loss = F.cross_entropy(activities_scores, activities_in)
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
                actions_loss = F.cross_entropy(actions_scores, actions_in, weight=actions_weights)
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


def train_volleyball(logger, data_loader, model, device, optimizer, epoch, cfg):
    # define metrics

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
            batch_video = batch_data_test[4];
            batch_clip = batch_data_test[5];
            # 第一个帧的所有batch的action id 向量
            actions_in = actions_in[:, 0, :].reshape((batch_size * cfg.num_boxes))
            activities_in = activities_in[:, 0].reshape((batch_size,))

            # forward
            # 得到个体分数和群体行为分数
            ret = model(batch_data_test[0], batch_data_test[1])
            # action_scores shape:torch.Size([48, 9])
            # activities scores shape:torch.Size([4, 8])

            loss_list = []

            # Predict activities
            if 'activities' in list(ret.keys()):
                activities_scores = ret['activities']
                activities_loss = F.cross_entropy(activities_scores, activities_in)
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
                actions_loss = F.cross_entropy(actions_scores, actions_in, weight=actions_weights)
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
}
