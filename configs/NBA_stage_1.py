from configs.config import Config


def NBA_stage_1():
    cfg = Config('NBA')
    cfg.data_path = 'data/NBA/videos'
    cfg.num_activities = 9
    cfg.use_gpu = True
    cfg.training_stage = 1
    cfg.stage1_model_path = 'result/basemodel_NBA_res18.pth'
    cfg.train_backbone = True
    cfg.test_before_train = False

    cfg.backbone = 'res18'
    cfg.image_size = 720, 1280
    cfg.out_size = 23, 40
    cfg.emb_features = 512
    cfg.add_figure = False


    cfg.batch_size = 2
    cfg.test_batch_size = 1
    cfg.num_frames = 1
    cfg.num_boxes = 14
    cfg.num_segments = 16

    cfg.train_learning_rate = 1e-4
    cfg.lr_plan = {30: 5e-5, 60: 2e-5, 90: 1e-5}
    cfg.max_epoch = 120
    cfg.set_bn_eval = False
    cfg.actions_weights = [[1., 1., 2., 3., 1., 2., 2., 0.2, 1.]]

    cfg.exp_note = 'NBA_stage1'
    return cfg
