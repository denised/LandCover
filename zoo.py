import numpy as np
import torch
import torch.nn
from fastai.layers import conv_layer
from fastai.basic_data import DataBunch
from fastai.basic_train import Learner
from fastai.core import ifnone, defaults
from fastai.vision import models, unet_learner
#from multispectral import coords
from multispectral import windows
from multispectral import corine

"""Models for working with Landsat / Corine data"""

################################################
# Convenience functions to create a torch dataset from a windows list
#

def dataset(windowlist: windows.WindowList) -> torch.utils.data.dataset.Dataset:
    """Convert a windowlist into a windowed dataset for Corine"""
    return windows.WindowedDataset(windowlist, corine.corine_labeler, *corine.corine_attributes())

def rgb_dataset(windowlist: windows.WindowList) -> torch.utils.data.dataset.Dataset:
    """Convert a windowlist into a windowed dataset with RGB channels only (discarding others)"""
    def rgb_label(lsdat, region):
        x,y = corine.corine_labeler(lsdat,region)
        # landsat band ordering is bgr, not rgb, so we have to reorder them as well
        xrgb = np.stack([x[2],x[1],x[0]])
        return xrgb,y
    return windows.WindowedDataset(windowlist, rgb_label, *corine.corine_attributes())


################################################
# Models
#

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
                              batch_size=bs)
        return learner


class ImageUResNet():
    """Use a classic imagenet trained Unet-resnet on only the RGB channels of the input data"""
    # This is really a thin wrapper around fastai's unet_learner function that lets me set a few defaults and record choices.
        
    @classmethod
    def learner(cls, tr_data, val_data, arch=models.resnet18, batch_size=None, loss_func=None, metrics=None, path=None, **kwargs) -> Learner:
        bs = ifnone(batch_size, defaults.batch_size)
        loss_func = ifnone(loss_func, defaults.loss_func)
        metrics = ifnone(metrics, defaults.metrics)
        path = ifnone(path, defaults.model_directory)
        databunch = DataBunch(rgb_dataset(tr_data).as_loader(bs=bs), 
                              rgb_dataset(val_data).as_loader(bs=bs))
        learner = unet_learner(databunch, arch, path=path, loss_func=loss_func, metrics=metrics, **kwargs)
        learner.params = dict(arch=arch, batch_size=bs, loss_func=loss_func, **kwargs)
        return learner