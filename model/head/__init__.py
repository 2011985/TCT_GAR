from .st_plus_tr_cross_cluster0 import ST_plus_TR_cross_cluster
from .global_sttr import global_sttr
from .pose_head2 import PoseAttent
from .pose_head3 import PoseAttent2
model_zoo = {
    'global_sttr': global_sttr,
    'st_plus_tr_cross_cluster':ST_plus_TR_cross_cluster,
    'pose_attent': PoseAttent,
    'pose_attent2':PoseAttent2
}

def HeadBuilder(config):
    model = model_zoo[config.type.lower()](config)
    return model
