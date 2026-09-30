from configs.config import Config


def volleyball_arg_infer():
    cfg = Config('volleyball')
    cfg.inference_module_name = 'arg_volleyball'
    cfg.data_path = 'data/volleyball'
    cfg.use_gpu = True
    cfg.training_stage = 2
    cfg.train_backbone = True
    cfg.test_before_train = True
    cfg.test_interval_epoch = 1

    # res18 setup
    cfg.backbone = 'res18'
    cfg.stage1_model_path = 'result/basemodel_VD_res18.pth'
    cfg.out_size = 23, 40
    cfg.emb_features = 512

    cfg.batch_size = 2
    cfg.test_batch_size = 1
    cfg.num_frames = 3
    cfg.load_backbone_stage2 = True
    cfg.train_learning_rate = 1e-4
    cfg.lr_plan = {11: 3e-5, 21: 1e-5}
    cfg.max_epoch = 30
    cfg.actions_weights = [[1., 1., 2., 3., 1., 2., 2., 0.2, 1.]]

    cfg.exp_note = 'ARG Volleyball_stage2'
    return cfg
