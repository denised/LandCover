import torch
from torch.tensor import Tensor
from fastai.core import defaults

def set_defaults(**overrides):
    """Set and add standard default settings, with any overrides specified by user"""
    defaults.__dict__.update(
        batch_size = 4,
        loss_func = torch.nn.BCEWithLogitsLoss(),
        metrics = [],
        model_directory = 'models',
        default_init = True
    )
    if len(overrides):
        defaults.__dict__.update(**overrides)

# only do an automatic set_defaults the first time this file is loaded
if not hasattr(defaults,'default_init'):
    set_defaults()
