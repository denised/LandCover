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
    def create(cls, tr_data, val_data, channels=(6,25,11), conv_size=None, bs=None, **kwargs):
        """Create a learner with defaults."""
        l_args, d_args = cls._init_args(**kwargs)
   
        bs = ifnone(bs, defaults.batch_size)
        tr_ds = cls.create_dataset(tr_data)
        val_ds = cls.create_dataset(val_data)
        databunch = DataBunch(tr_ds.as_loader(bs=bs), val_ds.as_loader(bs=bs), **d_args)
        arch_description = f"{cls.__name__} channels={channels} conv_size={conv_size or 'default'}"

        model = cls.create_model(channels, conv_size)

        learner = Learner(databunch, model, **l_args)
        learner.__class__ = cls
        learner.init_tracking(arch=arch_description)  # pylint: disable=no-member
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
    def create(cls, tr_data, val_data, arch=vision.models.resnet18, bs=None, **kwargs):
        l_args, d_args = cls._init_args(**kwargs)

        bs = ifnone(bs, defaults.batch_size)
        tr_ds = cls.create_dataset(tr_data)
        val_ds = cls.create_dataset(val_data)
        databunch = DataBunch(tr_ds.as_loader(bs=bs), val_ds.as_loader(bs=bs), **d_args)
        arch_description = f"{cls.__name__} arch={arch}"

        learner = vision.unet_learner(databunch, arch, **l_args)
        learner.__class__ = cls
        learner.init_tracking(arch=arch_description)  # pylint: disable=no-member
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
    def create(cls, tr_data, val_data, arch='resnet18', bs=None, **kwargs):
        l_args, d_args = cls._init_args(**kwargs)

        bs = ifnone(bs, defaults.batch_size)
        tr_ds = cls.create_dataset(tr_data)
        val_ds = cls.create_dataset(val_data)
        databunch = DataBunch(tr_ds.as_loader(bs=bs), val_ds.as_loader(bs=bs), **d_args)
        arch_description = f"{cls.__name__} arch={arch}"

        genfn = lambda _, arch=arch : cls.create_resnet(arch)
        learner = vision.unet_learner(databunch, genfn, pretrained=False, **l_args)
        learner.__class__ = cls
        learner.init_tracking(arch=arch_description)  # pylint: disable=no-member
        return learner


class MultiResClassifier(MultiUResNet):
    """Build the same way we build a MultiUResNet, but build a classifier instead.  So no UNet, and 
    change the target to be multi-label classification (presence or absence of each of the land use types)"""

    @classmethod
    def create_resnet(cls, arch='resnet18'):
        (block, layers) = cls.standard_arches[arch]
        model = ResNet(block, layers, num_classes=len(bands.CORINE_BANDS)-1)   # <-- Add num_classes here.
        model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    
    @classmethod 
    def create_dataset(cls, input_data):
        return windows.WindowedDataset(input_data, corine.corine_classifier, *corine.corine_attributes())
    
    @classmethod
    def create(cls, tr_data, val_data, arch='resnet18', bs=None, **kwargs):
        l_args, d_args = cls._init_args(**kwargs)

        bs = ifnone(bs, defaults.batch_size)
        tr_ds = cls.create_dataset(tr_data)
        val_ds = cls.create_dataset(val_data)
        databunch = DataBunch(tr_ds.as_loader(bs=bs), val_ds.as_loader(bs=bs), **d_args)
        arch_description = f"{cls.__name__} arch={arch}"

        model = cls.create_resnet(arch)
        
        learner = Learner(databunch, model, **l_args)
        learner.__class__ = cls
        learner.init_tracking(arch=arch_description)  # pylint: disable=no-member
        return learner