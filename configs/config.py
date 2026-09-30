import time
import os


class Config(object):
    """
    class to save config parameter
    """

    def __init__(self, dataset_name):
        # Global
        self.image_size = 720, 1280  # input image size
        self.batch_size = 16  # train batch size
        self.test_batch_size = 8  # test batch size
        self.num_boxes = 12  # max number of bounding boxes in each frame
        self.num_joints = 17

        # Gpu
        self.use_gpu = True
        self.use_multi_gpu = True
        self.device_list = "0,2,3"  # id list of gpus used for training

        # Dataset
        assert (dataset_name in ['volleyball', 'collective','NBA'])
        self.dataset_name = dataset_name

        if dataset_name == 'volleyball':
            self.data_path = 'data/volleyball/'  # data path for the volleyball dataload
            self.train_seqs = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53,
                               54, 0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]  # video id list of train set
            self.test_seqs = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]  # video id list of test set
        elif dataset_name == 'NBA':
            self.data_path = 'data/NBA/'
            self.train_seqs = [21801058,21801127,21801017,21801211,21801043,21800980,21801213,21801110,21801220,21801141,21801107,21801086,21801051,21801155,21800995,21801111,21800991,21801217,21801095,21801218,21801150,21801044,21801121,21800960,21801172,21801039,21801054,21801171,21801015,21801124,21801037,21801144,21801098,21801013,21801019,21801008,21801004,21801106,21800934,21801115,21800999,21801149,21801224,21800972,21801108,21801072,21801161,21801065,21801090,21800979,21800919,21800994,21801053,21801114,21801151,21801216,21801074,21800938,21801168,21800909,21801012,21800929,21801135,21800982,21801160,21800949,21801228,21801097,21801226,21800952,21801140,21800983,21801052,21801061,21800997,21800975,21801126,21801189,21801064,21801125,21801209,21800987,21801089,21801163,21800992,21801219,21801136,21800976,21801128,21801068,21800966,21801139,21800981,21801011,21801085,21801116,21801221,21801119,21801214,21801048,21800989,21801113,21800985,21801158,21800964,21801215,21801102,21801046,21800984,21801091,21801164,21801002,21801056,21801175,21801167,21801069,21801087,21801060,21801100,21800963,21800973,21800971,21801112,21801018,21801157,21801003,21801006,21801067,21801001,21801147,21801094,21801057,21801230,21801131,21801154,21801042,21801156,21801225,21801045,21801104,21800990,21801210,21801014,21801145,21801148,21801099,21800996,21801223,21801088,21801134,21800968]
            self.test_seqs = [21800974,21801076,21801079,21800970,21800965,21801165,21801071,21801049,21801070,21800977,21801162,21801120,21801007,21801078,21801123,21801188,21801038,21800978,21801159,21801152,21801204,21801229,21800988,21801077,21801153,21801063,21801096,21801105,21801050,21801129]
        else:
            self.data_path = 'data/collective'  # data path for the collective dataload
            self.test_seqs = [5, 6, 7, 8, 9, 10, 11, 15, 16, 25, 28, 29]
            self.train_seqs = [s for s in range(1, 45) if s not in self.test_seqs]

        # Backbone 
        self.backbone = 'res18'
        self.crop_size = 5, 5  # crop size of roi align
        self.train_backbone = False  # if freeze the feature extraction part of network, True for stage 1, False for stage 2
        self.out_size = 87, 157  # output feature map size of backbone
        self.emb_features = 1056  # output feature map channel of backbone

        # Activity Action
        self.num_actions = 9  # number of action categories
        self.num_activities = 8  # number of activity categories
        self.actions_loss_weight = 1.0  # weight used to balance action loss and activity loss
        self.actions_weights = None

        # Sample
        self.num_frames = 3
        self.num_before = 5
        self.num_after = 4

        # ARG params
        self.num_features_boxes = 1024
        self.num_features_relation = 256
        self.num_graph = 16  # number of graphs
        self.num_features_gcn = self.num_features_boxes
        self.gcn_layers = 1  # number of GCN layers
        self.tau_sqrt = False
        self.pos_threshold = 0.2  # distance mask threshold in position relation

        # Training Parameters
        self.train_random_seed = 0
        self.train_learning_rate = 1e-4  # initial learning rate
        self.lr_plan = {11: 3e-5, 21: 1e-5}  # change learning rate in these epochs
        self.train_dropout_prob = 0.3  # dropout probability
        self.weight_decay = 0  # l2 weight decay

        self.max_epoch = 30  # max training epoch
        self.test_interval_epoch = 1

        # Exp
        self.training_stage = 1  # specify stage1 or stage2
        self.stage1_model_path = ''  # path of the base model, need to be set in stage2
        self.test_before_train = False
        self.exp_note = 'Group-Activity-Recognition'
        self.exp_name = None
        self.set_bn_eval = False
        self.inference_module_name = 'dynamic_volleyball'
        self.add_figure = False

        # Dynamic Inference
        self.stride = 1
        self.ST_kernel_size = 3
        self.dynamic_sampling = True
        self.sampling_ratio = [1, 3]  # [1,2,4]
        self.group = 1
        self.scale_factor = True
        self.beta_factor = True
        self.load_backbone_stage2 = False
        self.parallel_inference = False
        self.hierarchical_inference = False
        self.lite_dim = None
        self.num_DIM = 1
        self.load_stage2model = False
        self.stage2model = None

        # Actor Transformer
        self.temporal_pooled_first = False

        # SACRF + BiUTE
        self.halting_penalty = 0.0001

    def init_config(self, rank, need_new_folder=True):
        # make the exp name
        if self.exp_name is None:
            time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            self.exp_name = '[%s_stage%d]<%s>' % (self.exp_note, self.training_stage, time_str)

        # make exp dir
        self.result_path = './result/%s' % self.exp_name
        self.log_path = './result/%s/log.txt' % self.exp_name

        if rank == 0 and need_new_folder:
            os.mkdir(self.result_path)
