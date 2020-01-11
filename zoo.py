import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import segmentation_models_pytorch as smp
from fastai.basics import *
from multispectral import windows
from multispectral import corine
from multispectral import bands
from infra import *
from typing import *

"""Models for working with Landsat / Corine data"""

# ###################################################################################################################
# One of the ways we vary what we are doing is by changing how we interpret the data...

def segmented_ms_dataset(input_data:windows.WindowList)->Dataset:
    """Treat the data as a multispectral segmentation problem"""
    return windows.WindowedDataset(input_data, corine.corine_labeler, *corine.corine_attributes())

def classified_ms_dataset(input_data:windows.WindowList)->Dataset:
    """Treat the data as a multispectral classification problem"""
    return windows.WindowedDataset(input_data, corine.corine_classifier, *corine.corine_attributes())

def segmented_rgb_dataset(input_data:windows.WindowList)->Dataset:
    """Transform the input to rgb, use for segmentation"""
    def rgb_label(lsdat, region):
        x,y = corine.corine_labeler(lsdat,region)
        # landsat band ordering is bgr, not rgb, so we have to reorder them as well
        xrgb = np.stack([x[2],x[1],x[0]])
        return xrgb,y
    return windows.WindowedDataset(input_data, rgb_label, *corine.corine_attributes())

def classified_rgb_dataset(input_data:windows.WindowList)->Dataset:
    """Transform the input to rgb, use for classification"""
    def rgb_label(lsdat, region):
        x,y = corine.corine_classifier(lsdat,region)
        # landsat band ordering is bgr, not rgb, so we have to reorder them as well
        xrgb = np.stack([x[2],x[1],x[0]])
        return xrgb,y
    return windows.WindowedDataset(input_data, rgb_label, *corine.corine_attributes())

def segmented_ms_band_dataset(input_data:windows.WindowList, sbands:List[int])->Dataset:
    """Perform multispectral segmentation over a simplified target that has only the requested bands"""
    def band_label(lsdat,region,sb=sbands):
        x,y = corine.corine_labeler(lsdat,region)
        y = y[sb]
        return x,y
    return windows.WindowedDataset(input_data, band_label, *corine.corine_attributes())


# ###################################################################################################################
# 
# Models

class Simple(LearnerPlus):
    """A simple sequence of (convolution, ReLU) pairs.  No up or down scaling."""

    def create_model(self, channels=(6,25,11), conv_size=None):
        """Channels is a sequence specifying how many channels there should be in between each layer; begins with #inputs and ends with #outputs
        Conv_size is the kernel size of each convolution.  Defaults to 3 for all layers."""
        nlayers = len(channels)-1
        conv_size = ifnone(conv_size, [3]*nlayers)
        layers = [ conv_layer( channels[i], channels[i+1], conv_size[i], padding=1 ) for i in range(nlayers) ]
        model = torch.nn.Sequential(*layers)
        self.parameters['arch'] = f"{self.__class__.__name__} channels={channels} conv_size={conv_size or 'default'}"
        return model
    
    def create_dataset(self, input_data): 
        return segmented_ms_dataset(input_data)


   

class MResNetEncoder(smp.encoders.resnet.ResNetEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # replace the conv1 with a 6 channel input
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)


# Now we have to add metadata for this to smp's metadata pool
mresnet18 = smp.encoders.encoders['resnet18'].copy()
mresnet18['encoder'] = MResNetEncoder
smp.encoders.encoders.update({'mresnet18':mresnet18})


class SMPModel(LearnerPlus):
    """Wraps one of the segmentation models"""

    def create_model(self, encoder, decoder):
        nets = { 'unet' : smp.Unet,
                 'linknet': smp.Linknet,
                 'fpn': smp.FPN,
                 'pspnet': smp.PSPNet }
        net = nets[decoder]
        model = net(encoder, classes=len(bands.CORINE_BANDS), activation=None, encoder_weights=None, decoder_use_batchnorm=False)
        self.parameters['arch'] = f"{self.__class__.__name__} {encoder}/{decoder}"
        return model

    def create_dataset(self, input_data):
        return segmented_ms_dataset(input_data)
    



# does not include mask bit.. todo: add.
class_counts = [507, 339, 783, 600, 145, 799, 811, 609, 544, 358]
# weights calculated as follows:
# class_median = (max(class_counts) + min(class_counts)) / 2
# class_weights = class_median / class_counts
# plus added 0.25 for the mask bit just arbitrarily.
class_weights = torch.tensor( [0.25, 0.942801, 1.410029, 0.610473, 0.796667, 3.296552, 0.598248, 0.589396, 0.784893, 0.878676, 1.335196] )  # pylint: disable=not-callable
#                              mask   water     barren    grass     shrub     wetlands   farm     forest    urban     cloud     shadow

def weighted_mse(x,y):
    """weighted MSE.  We don't use the builtin one because it applies weights to the *samples*, where we need weights on the *classes*"""
    global class_weights
    class_weights = class_weights.to( x.device )
    # separate assignments are easier to debug
    er = (x - y)
    er2 = (er * er).mean(other_dimensions(er,1))
    wer2 = class_weights * er2
    return wer2.mean()


# class OptValidator(LearnerCallback):
#     # pylint: disable=arguments-differ
#     """See whether we are in fact improving the model _on the given data_"""

#     def on_step_end(self, last_input, last_target, last_loss, **kwargs):
#         with self.learn.pause_training():
#             with torch.no_grad():
#                 updated_loss = fastai.basic_train.loss_batch(
#                     self.learn.model,
#                     last_input,
#                     last_target,
#                     self.learn.loss_func
#                 )
#         import pdb; pdb.set_trace()
    
#     # maybe finish this some day.
