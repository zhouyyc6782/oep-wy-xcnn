from model import *
from loss import *

import torch.optim as optim
import torch.nn as nn

MODEL_LIST = {
        # "cnn_lda_0": CNN_LDA_0, 
        # "cnn_gga_0": CNN_GGA_0,
        "cnn_gga_1": CNN_GGA_1,
        "cnn_gga_1_zsym": CNN_GGA_1_zsym, 
        # "cnn_gga_2": CNN_GGA_2,
        # "cnn_gga_2_zsym": CNN_GGA_2_zsym, 
        # "cnn_gga_3": CNN_GGA_3,
        # "cnn_gga_3_zsym": CNN_GGA_3_zsym, 
        # "cnn_gga_5": CNN_GGA_5, 
        # "cnn_gga_5_zsym": CNN_GGA_5_zsym, 
        # "cnn_gga_7": CNN_GGA_7, 
        # "cnn_gga_7_zsym": CNN_GGA_7_zsym, 

        # "cnn_gga_1_sigmoid": CNN_GGA_1_sigmoid, 
        # "cnn_gga_2_tanh": CNN_GGA_2_tanh,
        # "cnn_gga_15_rddm_elu": CNN_GGA_15_rddm_elu, 
        # "dnn": DNN,
        # "cnn_lda_pyramid_9_30": CNN_LDA_pyramid_9_30, 
        # "cnn_rdg_1": CNN_RDG_1,
        # "cnn_rdg_1_zsym": CNN_RDG_1_zsym,
        }

OPTIM_LIST = {
        "sgd": optim.SGD, 
        "dafault": optim.SGD, 
        }

LOSS_FUNC_LIST = {
        "mseloss": nn.MSELoss,
        "rmseloss": RMSELoss,
        "wmseloss": WMSELoss, 
        "default": nn.MSELoss,
        "mseloss_zsym": MSELoss_zsym, 
        }

