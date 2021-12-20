from __future__ import division
import importlib
import torch
import torch.nn as nn


MODEL_LIST = {
        "cnn_lda_0": "CNN_LDA_0", 
        "cnn_gga_0": "CNN_GGA_0",
        "cnn_gga_1": "CNN_GGA_1",
        }


class ExtendModel(nn.Module):
    def __init__(self, model, extend_size, *args):
        super(ExtendModel, self).__init__()
        self.model = _get_model(model, *args)
        # _add_extended_fc_layer(self, extend_size)

    def load_model(self, load_path):
        self.model.load_state_dict(torch.load(load_path))

    def forward(self, x):
        x = self.model.forward(x)
        # x = self.fc_extend(x)
        return x


def _add_extended_fc_layer(model, extend_size):
    model.fc_extend = nn.Linear(1, extend_size)
    param_iter = iter(model.fc_extend.parameters())
    # param = param_iter.next()
    param = next(param_iter)
    param.data[:] = 0.
    param.data[extend_size // 2] = 1.
    param = param_iter.next()
    param.data[:] = 0.


def _get_model(model_name, *args):
    model = None
    try:
        model_id = MODEL_LIST[model_name]
        model_mod = importlib.import_module(".%s" % (model_id), package=__package__)
        model = getattr(model_mod, model_id)(*args)
    except KeyError:
        print("No model named %s" % (model_name.lower()))
        exit(1)
    else:
        pass
    return model

