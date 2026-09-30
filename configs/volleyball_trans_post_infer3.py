from configs.config import Config

from munch import DefaultMunch

def volleyball_trans_post_infer():
    cfg = Config('volleyball')
    cfg.inference_module_name = 'trans_volleyball'
    # cfg.inference_module_name = 'dynamic_volleyball'
    cfg.data_path = '/data3/2021/hht/data/volleyball/videos/'
    cfg.use_gpu = True
    cfg.training_stage = 2
    cfg.train_backbone = True
    cfg.test_before_train = False
    cfg.test_interval_epoch = 1

    # res18 setup
    cfg.backbone = 'res18'
    cfg.stage1_model_path = 'result/basemodel_VD_res18.pth'
    cfg.pose_model_path = 'result/stage2_epoch28_85.19%.pth'
    cfg.out_size = 23, 40
    cfg.emb_features = 512

    cfg.batch_size = 2
    cfg.test_batch_size = 1
    cfg.num_frames = 10
    cfg.load_backbone_stage2 = True
    cfg.load_stage2model = False
    # cfg.stage2model = "result/stage2_epoch11_93.19%.pth"
    cfg.stage2model = "result/[Trans Volleyball_stage2_KST_base_stage2]<2024-03-14_19-27-20>/stage2_epoch18_91.25%.pth"
    cfg.train_learning_rate = 1e-4
    cfg.lr_plan = {3: 3e-5, 11:1e-5}
    # cfg.lr_plan = {30: 5e-5, 60: 2e-5, 90: 1e-5}

    cfg.max_epoch = 30
    cfg.actions_weights = [1., 1., 2., 3., 1., 2., 2., 0.2, 1.]
    cfg.projection_batchnorm = False
    cfg.projection_dropout = 0
    cfg.joint_initial_feat_dim = 3

    # trans configs
    cfg.acy_head = DefaultMunch.fromDict({
        'type':'global_sttr',
        'token_num':16,
        'token_feature_dim':512,
        'token_inchannel': 512,
        'embed_channel':256,
        'Nhead':8,
        'dropout_porb':0.1,
        'num_encoder_layers':2
    })

    cfg.head = DefaultMunch.fromDict({
        'type': 'st_plus_tr_cross_cluster',
        'num_encoder_layers': 1,
        'num_decoder_layers': 1,
        'num_STTR_layers': 3,
        'total_size':{
            'training': 12,
            'val': 12
        },
        'window_size': 3,
        'Nhead': 8,
        'dropout_porb': 0.1,
        'input_features': 12544, #256*7*7
        'embed_features': 256,
        'actions_num_classes': 9,
        'activities_num_classes': 8,
        'nhead':2,
        'dim_feedforward': 1024,
        'dropout': 0.5,
        'activation': 'relu',
        'num_joints':17,
        'num_boxes':12,
        'num_frames':10
    })

    cfg.pose_head = DefaultMunch.fromDict({
        'type': 'pose_attent2',
        'num_encoder_layers': 1,
        'num_decoder_layers': 1,
        'num_STTR_layers': 3,
        'total_size':{
            'training': 12,
            'val': 12
        },
        'window_size': 3,
        'Nhead': 8,
        'dropout_porb': 0.1,
        'input_features': 12544, #256*7*7
        'embed_features': 256,
        'actions_num_classes': 9,
        'activities_num_classes': 8,
        'nhead':2,
        'dim_feedforward': 1024,
        'dropout': 0.5,
        'activation': 'relu',
        'num_joints':17,
        'num_boxes':12,
        'num_frames':10,
        'num_tokens': 16
    })
    cfg.pose_head_infer = DefaultMunch.fromDict({
        'type': 'pose_attent',
        'num_encoder_layers': 1,
        'num_decoder_layers': 1,
        'num_STTR_layers': 3,
        'total_size':{
            'training': 12,
            'val': 12
        },
        'window_size': 3,
        'Nhead': 8,
        'dropout_porb': 0.1,
        'input_features': 12544, #256*7*7
        'embed_features': 256,
        'actions_num_classes': 9,
        'activities_num_classes': 8,
        'nhead':2,
        'dim_feedforward': 1024,
        'dropout': 0.5,
        'activation': 'relu',
        'num_joints':17,
        'num_boxes':12,
        'num_frames':10,
        'num_tokens': 16,
        'num_features_boxes': 256,
        'num_features_relation': 256,
        'num_graph': 16,  # number of graphs
        'num_features_gcn': 256,
        'gcn_layers': 1,  # number of GCN layers
        'tau_sqrt': False,
        'pos_threshold': 0.2,
        'out_size': (23, 40)
    })
    cfg.spatial_encoding = True
    cfg.num_features_boxes = 256
    cfg.keypoints = '/data3/2021/hht/data/volleyball/joints/ann.txt'
    cfg.load_posemodel = False

    cfg.exp_note = 'Trans Volleyball_stage2_KST_base'
    return cfg
