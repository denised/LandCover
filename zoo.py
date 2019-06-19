import torch
import torch.nn
import fastai
from fastai.layers import conv_layer
from fastai.basic_data import DataBunch
from fastai.basic_train import Learner
from fastai.core import ifnone, defaults
from neptunecontrib.monitoring.fastai import NeptuneMonitor
#from multispectral import coords
from multispectral.windows import WindowedDataset
from multispectral import corine
import neptune

        
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
            self.exp = neptune.create_experiment(params = self.learner.params)
    
    def on_train_end(self,**kwargs):
        if self.exp:
            self.exp.stop()
            self.exp = None


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

# only do an automatic set_defaults the first time it is loaded
if not hasattr(defaults,'default_init'):
    set_defaults()

"""Models that take 6-channel multispectral data in and 11-channel corine data as target."""

def dataset(windowlist):
    """Convert a windowlist into a windowed dataset for Corine"""
    return WindowedDataset(windowlist, corine.corine_labeler, *corine.corine_attributes())

class Simple(torch.nn.Sequential):
    """A simple sequence of (convolution, ReLU) pairs.  No up or down scaling."""
    # Note this is *almost* exactly thes same as fastai simple_cnn, but just not quite:
    # fastai simple_cnn has a flatten at the end, and we don't.
    def __init__(self, channels=(6,25,11), conv_size=None):
        """Channels is a sequence specifying how many channels there should be in between each layer; begins with #inputs and ends with #outputs
        Conv_size is the kernel size of each convolution.  Defaults to 3 for all layers."""
        nlayers = len(channels)-1
        conv_size = ifnone(conv_size, [3]*nlayers)
        layers = [ conv_layer( channels[i], channels[i+1], conv_size[i], padding=1 ) for i in range(nlayers) ]
        super().__init__(*layers)
        self.params = {
            'model' : 'Simple',
            'channel' : channels,
            'conv_size' : conv_size
        }
    
    @classmethod
    def learner(cls, tr_data, val_data, channels=(6,25,11), conv_size=None, batch_size=None, loss_func=None, metrics=None, path=None, **kwargs) -> Learner:
        """Create the fastai learner for the given data and simple model"""
        model = Simple(channels, conv_size)
        bs = ifnone(batch_size, defaults.batch_size)
        loss_func = ifnone(loss_func, defaults.loss_func)
        metrics = ifnone(metrics, defaults.metrics)
        path = ifnone(path, defaults.model_directory)
        databunch = DataBunch(dataset(tr_data).as_loader(bs=bs), dataset(val_data).as_loader(bs=bs))
        learner = Learner(databunch, model, path=path, loss_func=loss_func, metrics=metrics, **kwargs)
        learner.params = dict(model.params,
                              batch_size=bs,
                              loss_func=loss_func)
        return learner

