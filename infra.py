from contextlib import contextmanager
from pathlib import Path
import logging
import platform
import torch
from torch.tensor import Tensor
import fastai
from fastai.basic_data import DataBunch, DeviceDataLoader
from fastai.basic_train import Learner
from torch.utils.data.dataloader import DataLoader
from fastai.core import defaults, ifnone
from fastai.torch_core import rank_distrib, try_save
import neptune

_logger = logging.getLogger(__name__)

"""Some infrastructure to simplify model construction and training"""

###############################################
# Neptune integration
#
# Automatically set up netpune experiments and log training parameters on each call to
# learner.fit, using a fastai callback mechanism.  
# 
# Note: this is equivalent to neptune_contrib NeptuneMonitor, with the addition of auto
# initialization of experiments, and the fact that it will continue to work even if neptune
# isn't set up.
# 
        
class SetUpNeptune(fastai.basic_train.Callback):
    def __init__(self, learner):
        self.learner = learner
        self.exp = None
        self.own_exp = False  # if true, we created the experiment, so we should stop it

    def on_train_begin(self,**kwargs):
        try:  # if the caller has already created an experiment, don't override it.
            self.exp = neptune.get_experiment()
        except neptune.exceptions.NoExperimentContext:
            pass
        
        # this code could be inside the except body above; moved it outside for debugging
        if self.exp is None:
            # we normally expect to end up here.
            params = getattr(self.learner, 'params', {})
            params.update(
                opt=self.learner.opt,
                loss_func=self.learner.loss_func,
                callbacks=self.learner.callbacks,
                clstype=type(self.learner),
                machine=platform.node()
            )
            title = getattr(self.learner, 'title', "")
            try:
                self.exp = neptune.create_experiment(name=title, params=params)
                self.own_exp = True
            except neptune.exceptions.Uninitialized:
                _logger.warn("Neptune not initialized; no tracing will be done")
    
    def on_train_end(self,**kwargs):
        if self.exp and self.own_exp:
            self.exp.stop()
        self.exp = None
        self.own_exp = False

    def on_epoch_end(self, **kwargs):
        if self.exp:
            # fastai puts the validation loss as the first item in last_metrics
            metric_names = ['validation_loss'] + kwargs['metrics']
            for metric_name, metric_value in zip(metric_names, kwargs['last_metrics']):
                if metric_value is not None:
                    metric_name = getattr(metric_name, '__name__', metric_name)
                    self.exp.send_metric(str(metric_name), float(metric_value))
                else:
                    _logger.warn("{} has None value".format(metric_name))

    # def on_batch_end(self, **kwargs):
    #     if self.exp:
    #         self.exp.send_metric('training_loss', float(kwargs['last_loss']))

################################################
# Default settings handling.
# 
# This builds on the defaults feature of fastai (which isn't documented AFAIK)
#

def mse_sigmoid(y_pred:Tensor, y_true:Tensor):
    "Compute MSE after sigmoid"
    y_pred = y_pred.sigmoid()
    return fastai.metrics.mean_squared_error(y_pred, y_true)

def set_defaults(**overrides):
    """Set and add standard default settings, with any overrides specified by user"""
    defaults.__dict__.update(
        batch_size = 8,
        metrics = [mse_sigmoid],
        loss_func = torch.nn.BCEWithLogitsLoss(),
        model_directory = 'models',
        extra_callback_fns = [SetUpNeptune],
        default_init = True
    )
    if len(overrides):
        defaults.__dict__.update(**overrides)

# only do an automatic set_defaults the first time this file is loaded
if not hasattr(defaults,'default_init'):
    set_defaults()

################################################
# A couple changes to Learner --- override import/export and save params for Neptune

class LearnerPlus(Learner):
    
    # No implementation of __init__.  We don't need/use it given how we implement create.
    
    def export(self, file='export.pkl', destroy=False):
        "Export the state of the `Learner` to path"
        if rank_distrib(): return # don't save if slave proc
        args = ['opt_func', 'loss_func', 'metrics', 'true_wd', 'bn_wd', 'wd', 'train_bn', 'callback_fns', 'params']
        state = {a:getattr(self,a) for a in args}
        state['cb_state'] = {cb.__class__:cb.get_state() for cb in self.callbacks}
        state['model'] = self.model
        # No data export (for now)
        state['cls'] = self.__class__
        try_save(state, self.path, file)
        if destroy: self.destroy()
    
    @classmethod
    def create_from_file(cls, path, tr_data=None, val_data=None):
        "Load the learner object saved to path.  Optionally also set up data for it"
        state = torch.load(path, map_location='cpu') if defaults.device == torch.device('cpu') else torch.load(path)
        model = state.pop('model')
        cb_state = state.pop('cb_state')
        clas_func = state.pop('cls')
        params = state.pop('params', {})
        data = DummyDataBunch()
        res = clas_func(data, model, **state)
        res.callback_fns = state['callback_fns'] #to avoid duplicates
        res.callbacks = [fastai.basic_train.load_callback(c,s, res) for c,s in cb_state.items()]
        res.params = params
        if not ((tr_data is None) or (val_data is None)):
            res.set_data(tr_data, val_data)
        return res
    
    @classmethod
    def create_dataset(cls, input_data):
        """Specify a default way of generating a dataset for 'raw' data.  In our case, this will be a WindowList, but there is nothing requiring that."""
        # Gah.  Rearchitect.
        raise NotImplementedError
    
    def set_data(self, tr_data, val_data, bs=None):
        """Set data sources for this learner.  There is nothng magic in this method; it is just syntactic sugar to make the common case simpler."""
        tr_ds = self.create_dataset(tr_data)
        val_ds = self.create_dataset(val_data)
        bs = ifnone(bs, defaults.batch_size)
        self.data = DataBunch(tr_ds.as_loader(bs=bs), val_ds.as_loader(bs=bs))
    
    @contextmanager
    def temporary_validation_set(self, x_dataset):
        """Override the validation set for the duration of this context.  This is usually done to perform validation on a specific set of data."""
        stashed_validation_loader = self.data.valid_dl
        self.data.valid_dl = DeviceDataLoader(x_dataset.as_loader(), self.data.device)
        yield True
        self.data.valid_dl = stashed_validation_loader


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

################################################
# 