import numpy as np
import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from fastai.basics import *
from fastai import vision
from multispectral import windows
from multispectral import corine
from multispectral import bands
from infra import *

"""Models for working with Landsat / Corine data"""


class Simple(LearnerPlus):
    """A simple sequence of (convolution, ReLU) pairs.  No up or down scaling."""
    # Note this is *almost* exactly thes same as fastai simple_cnn, but just not quite:
    # fastai simple_cnn has a flatten at the end, and we don't.
   
    @classmethod
    def create_model(cls, channels=(6,25,11), conv_size=None):
        """Channels is a sequence specifying how many channels there should be in between each layer; begins with #inputs and ends with #outputs
        Conv_size is the kernel size of each convolution.  Defaults to 3 for all layers."""
        nlayers = len(channels)-1
        conv_size = ifnone(conv_size, [3]*nlayers)
        layers = [ conv_layer( channels[i], channels[i+1], conv_size[i], padding=1 ) for i in range(nlayers) ]
        model = torch.nn.Sequential(*layers)
        return model
    
    @classmethod
    def create_dataset(cls, input_data):
        """Create a dataset from a WindowList"""
        return windows.WindowedDataset(input_data, corine.corine_labeler, *corine.corine_attributes())
       
    @classmethod
    def create(cls, tr_data, val_data, channels=(6,25,11), conv_size=None, title="<untitled>", bs=None, **kwargs):
        """Create a learner with defaults."""
        l_args, d_args = cls._init_args(**kwargs)
   
        bs = ifnone(bs, defaults.batch_size)
        tr_ds = cls.create_dataset(tr_data)
        val_ds = cls.create_dataset(val_data)
        databunch = DataBunch(tr_ds.as_loader(bs=bs), val_ds.as_loader(bs=bs), **d_args)

        model = cls.create_model(channels, conv_size)

        learner = Learner(databunch, model, **l_args)
        learner.params = dict(channels=channels, conv_size=conv_size, bs=bs)
        learner.title = title
        learner.__class__ = cls
        return learner
    

class ImageUResNet(LearnerPlus):
    """Use a classic imagenet trained Unet-resnet on only the RGB channels of the input data"""
    # This is really a thin wrapper around fastai's unet_learner function that lets me set a few defaults and record choices.

    @classmethod
    def create_dataset(cls, input_data):
        """Convert a windowlist into a windowed dataset with RGB channels only (discarding others)"""
        def rgb_label(lsdat, region):
            x,y = corine.corine_labeler(lsdat,region)
            # landsat band ordering is bgr, not rgb, so we have to reorder them as well
            xrgb = np.stack([x[2],x[1],x[0]])
            return xrgb,y
        return windows.WindowedDataset(input_data, rgb_label, *corine.corine_attributes())
    
    @classmethod
    def create(cls, tr_data, val_data, arch=vision.models.resnet18, title="<untitled>", bs=None, **kwargs):
        l_args, d_args = cls._init_args(**kwargs)

        bs = ifnone(bs, defaults.batch_size)
        tr_ds = cls.create_dataset(tr_data)
        val_ds = cls.create_dataset(val_data)
        databunch = DataBunch(tr_ds.as_loader(bs=bs), val_ds.as_loader(bs=bs), **d_args)

        learner = vision.unet_learner(databunch, arch, **l_args)
        learner.params = dict(arch=arch, bs=bs)
        learner.title = title
        learner.__class__ = cls
        return learner


class MultiUResNet(LearnerPlus):
    """The same as UNet-ResNet, but accepting 6 input bands instead of 3"""
    
    standard_arches = {
        'resnet18': (BasicBlock, [2, 2, 2, 2]),
        'resnet34': (BasicBlock, [3, 4, 6, 3]),
        'resnet50': (Bottleneck, [3, 4, 6, 3]),
        'resnet101': (Bottleneck, [3, 4, 23, 3]),
        'resnet152': (Bottleneck, [3, 8, 36, 3])
    }

    @classmethod
    def create_resnet(cls, arch='resnet18'):
        # Literally the only difference with a standard resnet is that the initial layer takes 6 channel
        # input instead of 3.  So we create the model by creating a standard resnet then swapping out
        # the first layer for the one we want.
        (block, layers) = cls.standard_arches[arch]
        model = ResNet(block, layers)
        model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    
    @classmethod
    def create_dataset(cls, input_data):
        """Create a dataset from a WindowList"""
        return windows.WindowedDataset(input_data, corine.corine_labeler, *corine.corine_attributes())

    @classmethod
    def create(cls, tr_data, val_data, arch='resnet18', title="<untitled>", bs=None, **kwargs):
        l_args, d_args = cls._init_args(**kwargs)

        bs = ifnone(bs, defaults.batch_size)
        tr_ds = cls.create_dataset(tr_data)
        val_ds = cls.create_dataset(val_data)
        databunch = DataBunch(tr_ds.as_loader(bs=bs), val_ds.as_loader(bs=bs), **d_args)

        genfn = lambda _, arch=arch : cls.create_resnet(arch)
        learner = vision.unet_learner(databunch, genfn, pretrained=False, **l_args)
        learner.params = dict(arch=arch, bs=bs)
        learner.title = title
        learner.__class__ = cls
        return learner
    

class CorineDataStats(Callback):
    """Put some very corine-specific stats about the data into metrics."""
    _order = -20 #Needs to run before other callbacks that will look at metric_names
    stats = [0] * 5

    def on_train_begin(self, metrics_names, **kwargs):
        # return a set of classes we are going to look at.  #x will count how many x occur in the target data,
        # while learnedany will aggregate over all of them in the learned output
        self.names = ["#barren","#urban","#wetlands","#shrub","learnedany"]
        # indexes of these classes in the data
        self.indexes = [ bands.band_index(bands.CORINE_BANDS,b) for b in ['barren','urban','wetlands','shrub'] ]
        # Modifying metrics_names doesn't work in a regular callback but it does in CycleHandler
        return { 'metrics_names' : metrics_names +  self.names }
    
    def on_epoch_begin(self, **kwargs):
        self.stats = [0] * 5
    
    def on_batch_end(self, last_target, last_output, train, **kwargs):
        if train:
            last_target = last_target.detach()    # dunno if we need to detach or not, but it doesn't hurt
            last_output = last_output.detach().sigmoid()   # shouldn't have to know to put sigmoid() here!

            for i in range(len(self.indexes)):
                # Note we are summing here, which will get us a density measure rather than an actual count
                # (since the values are floats, not necessarily 0..1)
                self.stats[i] += last_target[:,self.indexes[i]].sum().item()
                
                # But for learning, we're only going to sum high-value items, so it is more count-like
                lo = last_output[:,self.indexes[i]]
                biglo = lo.ge(0.7)   # in numpy, this would be lo[lo>0.7].sum()
                self.stats[4] += torch.masked_select(lo,biglo).sum()
    
    def on_epoch_end(self, num_batch, last_metrics, last_target, **kwargs):
        stats = [ 0.0 for x in self.stats ]
        try: # protect divide by zero or anything else that might happen.
            size = num_batch * last_target.numel() // last_target.size()[1]
            stats = [ math.floor( 100 * x / size ) for x in self.stats ]
        except:
            pass

        return { 'last_metrics' : last_metrics + stats }
        
    # For reasons I don't understand, Callback __repr__ breaks on this class, so overriding here...
    # Somehow (perhaps because Jupyter/IPython?), the methods get wrapped, and the call to func_args()
    # fails with the error  AttributeError: 'method-wrapper' object has no attribute '__code__'
    def __repr__(self):
        return "zoo.CorineDataStats()"


def standard_monitor(n=100):
    """Construct a cycle monitor with standard stuff in it"""
    cbs = [ CorineDataStats(), Validate, SendToNeptune ]
    return CycleHandler.create(n=n, callbacks = cbs)


class SumQuadLoss(nn.Module):
    def __init__(self, epsilon=0.01):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, input, target):
        # TODO: apply epsilon to target
        input2 = torch.sigmoid(input)
        return torch.pow( input2 - target, 4).sum()
