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
            class_index = 1,       # The index of a batch that represents class.  (Usually 1 or -1)
            train_end = None,      # If using TrainEnd callback, how many iterations to run before terminating
            trace_pdb = False,     # if using LearnerTracer callback and a trace event occurs, drop into the debugger
            clean_frequency = 100,  # If using LearnerCleaner callback, how often to clean up
            classnames = [  # copied from multispectral.bands so I don't have to cross-import
                'mask',   # 1 if data, 0 if none
                'water',
                'barren',
                'grass',
                'shrub',
                'wetlands',
                'forest',
                'farm',
                'urban',
                'cloud',
                'shadow'],
            default_init = True
        )
    if len(overrides):
        defaults.__dict__.update(**overrides)

# only do an automatic set_defaults the first time this file is loaded
if not hasattr(defaults,'default_init'):
    set_defaults()
