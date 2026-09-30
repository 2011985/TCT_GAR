from .volleyball_stage_1 import *
from .volleyball_arg_infer import *
from .volleyball_pctdm_infer import *
from .volleyball_sacrf_infer import *
from .NBA_stage_1 import *
from .NBA_stage_2_SAM import *
from .volleyball_trans_post_infer3 import *
from .volleyball_din_infer import *
def get_config(my_fig):
    return eval(my_fig)()
