import torch
from fastai.core import defaults
from .ranger import Ranger

def set_defaults(**overrides):
    """Set standard default settings, with any overrides specified by user.  When called with no
    arguments, resets defaults to original values."""
    # Specify our 'default defaults' only the first time, so as not to override anything the user has set.
    # Actually to completely reset defaults we'd have to go collect all the ones that appear in fastai...don't need this yet.
    if not hasattr(defaults,'default_init') or len(overrides) == 0:
        defaults.__dict__.update(
            batch_size = 4,
            loss_func = torch.nn.BCEWithLogitsLoss(),
            opt_func = Ranger,
            metrics = [],
            model_directory = 'models',
            callback_fns = [],
            default_init = True
        )
    if len(overrides):
        defaults.__dict__.update(**overrides)

# only do an automatic set_defaults the first time this file is loaded
if not hasattr(defaults,'default_init'):
    set_defaults()
