from contextlib import contextmanager
import torch
from fastai.basic_train import *
from fastai.basic_data import DataBunch, DeviceDataLoader
from torch.utils.data.dataloader import DataLoader
from fastai.core import defaults, ifnone
from .traintracker import TrainTracker

################################################
# Our augmentation of the fastai Learner class (and related classes)
# * Merge the creation of Learner and DataBunch
# * Make it easy to substitute new data into an existing Learner
# * Add context handlers for pausing training, and for using an alternate dataset
# * Support experiment tracking (via TrainTracker)

class LearnerPlus(Learner):

    def __init__(self, tr_data, val_data, **kwargs):
        learner_args, other_args = self._partition_args(**kwargs)

        # self.parameters is a bunch of metadata used for tracking.
        # see keys at the top of traintracker.py
        self.parameters = {
            "code_marker" : "add mask to input",  # hardwired description of significant code update
            "arch" : self.__class__.__name__  # may be overridden by subclasses to add more info
        }

        bs = defaults.batch_size
        tr_ds = self.create_dataset(tr_data)
        val_ds = self.create_dataset(val_data)
        databunch = DataBunch(tr_ds.as_loader(bs=bs), val_ds.as_loader(bs=bs))
        model = self.create_model(**other_args)
        TrainTracker.default_model_id(model)

        super().__init__(databunch, model, **learner_args)

        self.callback_fns.insert(0,TrainTracker)

  
    def create_model(self, **kwargs):
        """Subclasses must implement this function"""
        raise NotImplementedError

    def create_dataset(self, input_data):
        """Generate a dataset for the specified input data.  Subclasses must implement this function"""
        raise NotImplementedError

    @classmethod
    def _partition_args(cls, opt_func=None, loss_func=None, metrics=None, true_wd=True, bn_wd=True, wd=None, train_bn=True,
                        path=None, model_dir=None, callback_fns=None, callbacks=None, layer_groups=None, add_time=True, silent=None, 
                        **kwargs):
        """Pull out arguments for the learner class and set their defaults.  Returns a tuple (learner-args, other-args)"""
        # Why we do this seemingly pointless thing:  this allows learner __init__ to accept a mixed kwargs that covers Learner and model
        # Here we (a) separate the mixed kwargs into two lists, and (b) standardize the default values for the learner args (which include additional
        # defaults beyond what fastai does).
        learner_args = {
            'opt_func': ifnone(opt_func, defaults.opt_func),
            'loss_func': ifnone(loss_func, defaults.loss_func),
            'metrics': ifnone(metrics, defaults.metrics),
            'true_wd': true_wd,
            'bn_wd': bn_wd,
            'wd': ifnone(wd, defaults.wd),
            'train_bn': train_bn,
            'path': ifnone(path, defaults.model_directory),
            'model_dir': ifnone(model_dir, '.'),  # This is a change from fastai default: we mix leaner exports and models in same dir
            'callback_fns': ifnone(callback_fns, defaults.callback_fns),
            'callbacks': callbacks,
            'layer_groups': layer_groups,
            'add_time': add_time,
            'silent': silent
        }
        return (learner_args, kwargs)

    # ##############################  Save / Load weights
    # Save / load model weights only.  Allows for "loose" transfer of weights between different models
    def save_model_weights(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load_model_weights(self, path, strict=False):
        with self.pause_training():
            state = torch.load(path, map_location=self.device())
            self.model.load_state_dict(state, strict)

    # ##############################  Fit
    # Add a few details to fit to support tracking.

    def _pre_fit(self, kwargs):
        """Add a few extra keyword parameters that we will accept in fit, lr_find, etc."""
        # If a 'description' keyword is present, store it (handed off to TrainTracker)
        if 'description' in kwargs:
            self.parameters['description'] = kwargs['description']
            del kwargs['description']
        
        # If a 'neptune' keyword is present, that will control whether neptune records this run or not
        if 'neptune' in kwargs:
            self.do_neptune = kwargs['neptune']
            del kwargs['neptune']
        

    def fit(self, *args, **kwargs):  # pylint: disable=arguments-differ
        self._pre_fit(kwargs)
        self.parameters['epochs'] = args[0]
        if 'parameters' not in self.parameters: self.parameters['parameters'] = TrainTracker.describe_parameters(self,**kwargs)
        if 'invocation' not in self.parameters: self.parameters['invocation'] = "fit"       

        super().fit(*args, **kwargs)

        # clean up
        if hasattr(self,'do_neptune'): del self.do_neptune
        del self.parameters['epochs']
        del self.parameters['parameters']
        del self.parameters['invocation']
    
    def fit_one_cycle(self, *args, **kwargs):
        self._pre_fit(kwargs)
        self.parameters['invocation'] = "fit-one-cycle"
        super().fit_one_cycle(*args, **kwargs)  # pylint: disable=no-member
    
    def lr_find(self, *args, **kwargs):
        self._pre_fit(kwargs)
        self.parameters['invocation'] = "lr_find"
        super().lr_find(*args, **kwargs) # pylint: disable=no-member


    # ################################  Context Managers
    # Making a few situations easier...
       
    @contextmanager
    def temporary_validation_set(self, x_dataset, pause_train=True):
        """Override the validation set for the duration of this context.  This is usually done to perform evaluation of a specific set of data.
        If pause_train is true, the model is also set to eval for the duration (and restored to its previous setting at the end)."""
        stashed_train_state = self.model.training
        stashed_validation_loader = self.data.valid_dl
        self.data.valid_dl = DeviceDataLoader(x_dataset.as_loader(), self.data.device)
        if pause_train: self.model.eval()
        yield True
        self.data.valid_dl = stashed_validation_loader
        if pause_train: self.model.train(stashed_train_state)
    
    @contextmanager
    def pause_training(self):
        """Set model to eval temporarily.  Model is restored to initial state afterwards (which might have been either train or eval)."""
        stashed_train_state = self.model.training
        self.model.eval()
        yield True
        self.model.train(stashed_train_state)

    # ################################  Misc
    
    def device(self):
        """Return the device this learner's parameters are on.  (Assumes they are all on the same device...)"""
        ps = self.model.parameters()
        return next(ps).device
    
    def set_data(self, tr_data, val_data, bs=None):
        """Set data sources for this learner."""
        tr_ds = self.create_dataset(tr_data)
        val_ds = self.create_dataset(val_data)
        bs = ifnone(bs, defaults.batch_size)
        self.data = DataBunch(tr_ds.as_loader(bs=bs), val_ds.as_loader(bs=bs))
        if 'data' in self.parameters: del self.parameters['data']  # force recomputation

