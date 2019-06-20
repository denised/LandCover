import torch
import fastai
from fastai.core import defaults
from neptunecontrib.monitoring.fastai import NeptuneMonitor
import neptune

"""Some infrastructure to simplify model construction and training"""

###############################################
# Neptune integration
#
# Automatically set up netpune experiments and log training parameters on each call to
# learner.fit, using a fastai callback mechanism.
#
        
class SetUpNeptune(fastai.basic_train.Callback):
    # TODO: simply replace NeptuneMonitor with our own, so we can be (1) be
    # graceful about whether we are using neptune or not, and (2)
    # customize the data it sends back better.

    def __init__(self, learner, **kwargs):
        self.learner = learner
        self.exp = None

    def on_train_begin(self,**kwargs):
        try:  # if the caller has already created an experiment, don't override it.
            neptune.get_experiment()
        except neptune.exceptions.NoExperimentContext:
            # we normally expect to end up here.
            params = self.learner.params
            params.update(
                opt=self.learner.opt,
                loss_func=self.learner.loss_func,
                callbacks=self.learner.callbacks
            )
            self.exp = neptune.create_experiment(params = params)
    
    def on_train_end(self,**kwargs):
        if self.exp:
            self.exp.stop()
            self.exp = None

################################################
# Default settings handling.
# 
# This builds on the defaults feature of fastai (which isn't documented AFAIK)
#

def set_defaults(**overrides):
    """Set and add standard default settings, with any overrides specified by user"""
    defaults.__dict__.update(
        batch_size = 8,
        metrics = [fastai.metrics.mean_squared_error],
        loss_func = torch.nn.BCEWithLogitsLoss(),
        model_directory = '.',
        extra_callback_fns = [SetUpNeptune],
        extra_callbacks = [NeptuneMonitor()],
        default_init = True
    )
    if len(overrides):
        defaults.__dict__.update(**overrides)

# only do an automatic set_defaults the first time this file is loaded
if not hasattr(defaults,'default_init'):
    set_defaults()

