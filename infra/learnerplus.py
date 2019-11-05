from contextlib import contextmanager
from pathlib import Path
from matplotlib import pyplot
import torch
import fastai
from fastai.basic_train import *
from fastai.basic_data import DataBunch, DeviceDataLoader
from torch.utils.data.dataloader import DataLoader
from fastai.core import defaults, ifnone
from fastai.torch_core import rank_distrib, try_save
from .traintracker import TrainTracker

################################################
# Our augmentation of the fastai Learner class (and related classes)
# * Merge the creation of Learner and DataBunch
# * Make it easy to substitute new data into an existing Learner
# * Add context handlers for pausing training, and for using an alternate dataset
# * Support export/import of Learner state without Dataset
# * Support experiment tracking (via TrainTracker)

class LearnerPlus(Learner):
  
    # ##############################  Import / Export 
    # Our export/create_from_file are similar to fastai export/save/load, but with these differences:
    #   * we don't save or restore data (too large)
    #   * create_from_file allows for the possibility that the class of the learner is different than the class
    #     that was saved.
    # It would be nice if there was a way to do this without copying so much of the fastai code, but I couldn't
    # think of one.

    def export(self, file='export.pkl', destroy=False):
        "Export the state of the `Learner` to path"
        if rank_distrib(): return # don't save if slave proc
        args = ['opt_func', 'loss_func', 'metrics', 'true_wd', 'bn_wd', 'wd', 'train_bn', 'callback_fns', 'parameters']
        state = {a:getattr(self,a) for a in args}
        state['cb_state'] = {cb.__class__:cb.get_state() for cb in self.callbacks}
        state['model'] = self.model
        # No data export (for now)
        state['cls'] = self.__class__
        try_save(state, self.path, file)
        if destroy: self.destroy()
    
    # Create from file is like 
    @classmethod
    def create_from_file(cls, path, tr_data=None, val_data=None):
        "Load the learner object saved to path.  Optionally also set up data for it"
        state = torch.load(path, map_location='cpu') if defaults.device == torch.device('cpu') else torch.load(path)
        model = state.pop('model')
        cb_state = state.pop('cb_state')
        clas_func = state.pop('cls')
        params = state.pop('parameters', {})
        data = DummyDataBunch()
        res = clas_func(data, model, **state)
        res.callback_fns = state['callback_fns'] #to avoid duplicates
        res.callbacks = [fastai.basic_train.load_callback(c,s, res) for c,s in cb_state.items()]
        if 'data' in params: del params['data']
        res.parameters = params
        if not ((tr_data is None) or (val_data is None)):
            res.set_data(tr_data, val_data)
        return res

    # ##############################  Save / Load weights
    # Save / load model weights only.  Allows for "loose" transfer of weights between different models
    def save_model_weights(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load_model_weights(self, path, strict=False):
        with self.pause_training():
            state = torch.load(path, map_location=self.device())
            self.model.load_state_dict(state, strict)
    
    # ##############################  Initialization
    # Note we don't have an __init__ method.  We don't use it.  Instead each subclass is expected to define a @classmethod create method that creates
    # a learner object, together with it's data.  See the classes in zoo.py for examples.

    @classmethod
    def create_dataset(cls, input_data):
        """Specify a default way of generating a dataset for 'raw' data.  In our case, this will be a WindowList, but there is nothing requiring that."""
        # Gah.  Rearchitect.
        raise NotImplementedError
    
    def set_data(self, tr_data, val_data, bs=None):
        """Set data sources for this learner."""
        tr_ds = self.create_dataset(tr_data)
        val_ds = self.create_dataset(val_data)
        bs = ifnone(bs, defaults.batch_size)
        self.data = DataBunch(tr_ds.as_loader(bs=bs), val_ds.as_loader(bs=bs))
        if 'data' in self.parameters: del self.parameters['data']  # force recomputation
   
    @classmethod
    def _init_args(cls, opt_func=None, loss_func=None, metrics=None, true_wd=True, bn_wd=True, wd=None, train_bn=True,
                   path=None, model_dir=None, callback_fns=None, callbacks=None, layer_groups=None, add_time=True, silent=None, 
                   **kwargs):
        """Pull out arguments for the learner class and set their defaults.  Returns a tuple learner-args, other-args"""
        # Why we do this seemingly pointless thing:  this allows our Learner create methods to accept a mixed kwargs that covers both Learner and DataBunch
        # Here we (a) separate the mixed kwargs into two lists, and (b) standardize the default values for the learner args (which include additional
        # defaults beyond what fastai does)
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
        return learner_args, kwargs
    
    def init_tracking(self, **kwargs):
        """Initialize additional learner state specific to LearnerPlus.  kwargs adds to or overrides this state."""
        # See the documentation at the top of traintracker for the fields and their meanings:
        self.parameters = {
            "code_marker" : "classifier includes mask band",  # hardwired description of significant code update
        }
        self.parameters.update(kwargs)
        self.callback_fns.insert(0,TrainTracker)
        # default setting when we create a *new* model only
        # we set it so that model state is loaded, it can pick up the tracker id
        TrainTracker.set_model_id(self.model,"empty")

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
        """Return the device this learner's buffers are on.  (Assumes they are all on the same device...)"""
        # OK, I lied.  The first buffer is (or may be) a traintracker buffer, which is on the CPU.
        # Grab the one after that instead.
        # TODO: is there a way to make this more principled?
        bfs = self.model.buffers()
        skip = next(bfs)   # pylint: disable=unused-variable
        return next(bfs).device


class DummyDataBunch(DataBunch):
    def __init__(self):
        self.device = defaults.device
        self.path = Path(".")
        self.train_dl = DataLoader([])
        self.valid_dl = DataLoader([])
        self.fix_dl = None
        self.test_dl = None
    
    def __repr__(self):
        return "DummyDataBunch()"

# Simple utility for accumulating learner.recorder information.

class LRAccumulator(object):
    """Accumulate multiple recorder results to compare them on the same graph.  Can be applied across any Learner fit method
    (lr_find, fit, etc.), and a single accumulator can be used across multiple learners, models, data... anything where you'd like
    to compare the loss graphs."""
    def __init__(self, learner=None, title="a", fmt=''):
        """Create a new accumulator, optionally starting with an initial recorder trace."""
        self.curves = []
        if learner:
            self.add(learner, title, fmt)
 
    def add(self, learner, title=None, fmt=''):
        """Add another recorder trace to the list.
        The format of the curve can be specified with the fmt argument using the matplotlib format shortcut notation (e.g. 'ro-')"""
        title = ifnone(title, chr(ord("a") + len(self.curves)))
        self.curves.append( (title, learner.recorder.lrs, [x.item() for x in learner.recorder.losses], fmt) )
    
    def drop(self, index=-1):
        """Add the wrong curve by mistake?"""
        del self.curves[index]
    
    def plot(self, bylrs=True, xmin=None,xmax=None,ymin=None,ymax=None):
        """Plot all the accumulated curves.  By default, plots loss against learning rate (which is appropriate for comparing lr_find
        results).  To compare other loss traces, set `bylrs=False`.  By default the graph will be scaled to include all the data for
        all the curves; use the xmin/max and ymin/max arguments to focus on the interesting part."""
        _, ax = pyplot.subplots(1,1)
        for (label, xs, ys, fmt) in self.curves:
            if bylrs:
                ax.plot(xs, ys, fmt, label=label)
            else:
                ax.plot(ys, fmt, label=label)
        ax.set_ylabel("Loss")
        ax.set_xlabel("Learning Rate" if bylrs else "Batch")
        if xmin is not None or xmax is not None:
            ax.set_xlim(left=xmin, right=xmax)
        if ymin is not None or ymax is not None:
            ax.set_ylim(bottom=ymin, top=ymax)
        if bylrs: 
            ax.set_xscale('log')
            ax.xaxis.set_major_formatter(pyplot.FormatStrFormatter('%.0e'))
        ax.legend()
