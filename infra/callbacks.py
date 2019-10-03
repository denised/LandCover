import torch
import numpy as np
from fastai.basic_train import *
from fastai.core import defaults, ifnone
from fastai.callback import Callback
from functools import partial

# Some of these callbacks are meant to be used with CycleHandler.  Specifically, the way that metrics add themselves to the list of
# metrics is correct with CycleHandler, but not with Recorder

class Validate(LearnerCallback):
    """Run validation on demand, potentially with a list of callbacks.  The callbacks would most likely be used to do test time augmentation (TTA)."""

    def __init__(self, learn, callbacks=None):
        super().__init__(learn)
        self.callbacks = callbacks or []

    def on_train_begin(self, pbar, metrics_names, **kwargs):
        self.pbar = pbar
        metrics_names = ifnone(metrics_names, [])
        metrics_names.append('val_loss')
        return { 'metrics_names': metrics_names  }

    def on_epoch_end(self, **kwargs):
        #log_it(self, 'on_epoch_end', **kwargs)
        skip_validate = kwargs['skip_validate'] if 'skip_validate' in kwargs else False
        last_metrics = kwargs['last_metrics'] if 'last_metrics' in kwargs else []
        if skip_validate:
            last_metrics.append( None )
        else:
            with self.learn.pause_training():
                cbh = CallbackHandler(self.callbacks) if self.callbacks else None
                result = validate(self.learn.model, self.learn.data.valid_dl, loss_func=self.learn.loss_func, cb_handler=cbh, pbar=self.pbar)
                last_metrics.append( result )
        return { 'last_metrics' : last_metrics }

class LearnerTracer(LearnerCallback):
    """Print warnings for certain occurrances: forward data containing nans, or gradients that either contain nan or are all-zero."""
    def on_train_begin(self, **kwargs):
        self.hooks = []
        
        def hook(module, input, output, name):
            if module.training and torch.isnan(output).sum().item() > 0:
                print("warning: {} output: nan(s)".format(name))
                if defaults.trace_pdb:
                    import pdb; pdb.set_trace()

        for n, m in self.learn.model.named_modules():
            self.hooks.append( m.register_forward_hook( partial(hook, name=n) ))

    def on_backward_end(self, **kwargs):
        # Check the gradients
        # For better perf, first do a check of all grads together, and only if we find an issue do we check them individually.
        zeroifbad = [ (p.grad == p.grad).prod()   *   p.grad.ne(0).sum() for p in self.learn.model.parameters() if p.grad is not None ]
        #  zero if            some nan            or       all-zero

        if torch.stack( zeroifbad ).min().item() == 0:
            # check everything individually
            for name,p in self.learn.model.named_parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).sum().item() > 0:
                        print("warning: {} grad: nan(s)".format(name))
                    elif p.grad.ne(0).sum().item() == 0:
                        print("warning: {} grad: all zero".format(name))
            if defaults.trace_pdb:
                import pdb; pdb.set_trace()

    def on_train_end(self, **kwargs):
        for hook in self.hooks:
            hook.remove()

class GradientMetrics(LearnerCallback):
    """Report gradient magnitude means.  Gives a very rough estimate of how much the model is changing.
    Gradients are sampled at the epoch end (not accumulated over the epoch)."""
    def on_train_begin(self, metrics_names, **kwargs):
        return { 'metrics_names' : metrics_names +  ["min GMM", "mean GMM", "max GMM"] }
    
    def on_epoch_end(self, last_metrics, **kwargs):
        stats = np.array([ p.grad.abs().mean().item() for p in self.learn.model.parameters() if p.grad is not None])
        return { 'last_metrics': last_metrics + stats }


class TrainEnder(LearnerCallback):
    # Note: fastai has a callback just like this, but I like setting the train_end parameter via defaults instead of __init__.
    # Makes initializing it cleaner.
    def on_batch_end(self, num_batch, **kwargs):
        if defaults.train_end and num_batch >= defaults.train_end:
            return {'stop_epoch': True, 'stop_training': True, 'skip_validate': True}

    
class LearnerCleaner(LearnerCallback):
    _order = -10
    """Clean up some stuff (currently just the per-batch recorder data) every so many batches."""
    def on_batch_begin(self, num_batch, **kwargs):
        if defaults.clean_frequency and num_batch % defaults.clean_frequency == 0:
            self.learn.recorder.losses = []
            self.learn.recorder.lrs = []
            self.learn.recorder.moms = []


#############################################################################################################
#
# LOSS Functions
#
#############################################################################################################
#
# There are too many possible variations here...it is confusing
#    whether/which activation function (which depends on whether the model already does it and what the loss function expects)
#    packing multiple values into a single function (for reporting purposes)
#    the distinction between training (autograd=True) and reporting (autograd=False)
#    auto averaging across batches (or not)
#    form (function, nn.Module, Callback)
#
# And to this we want to add one more: epsilon shrinking of the target (moving values away from 0 and 1 endpoints).
#
# Fastai has some of this figured out, but not all of it.
# Figuring out the right architecture to do minimal code and graft cleanly onto what fastai already does...
# too hard for now.  These will work.

class TrainLoss(object):
    """Wrapper around a loss function to apply activation and or epsilon shrinking of target"""
    def __init__(self, fn, activation=torch.sigmoid, epsilon=None):
        self.loss_fn = fn
        self.activation = activation
        if callable(epsilon):
            self.epsilon = epsilon
        elif isinstance(epsilon, float):
            self.epsilon = lambda t, e=epsilon: e + (t * (1-2*e))
        else:
            assert epsilon is None
            self.epsilon = None
    
    def __call__(self, prediction, target):
        prediction2 = self.activation(prediction) if self.activation else prediction
        target2 = self.epsilon(target) if self.epsilon else target
        return self.loss_fn(prediction2, target2)


class ReportLoss(Callback):
    """Wrapper around loss function to apply activation and or epsilon shrinking of target.  
    ReportLoss is applied per-epoch only (use AverageMetric instead to accumulate)."""
    def __init__(self, fn, name, activation=torch.sigmoid, epsilon=None):
        super().__init__(self)
        self.loss_fn = fn
        self.name = name
        self.activation = activation
        if callable(epsilon):
            self.epsilon = epsilon
        elif isinstance(epsilon, float):
            self.epsilon = lambda t, e=epsilon: e + (t * (1-2*e))   # TODO: inplace version?, e.g. torch.add_
        else:
            assert epsilon is None
            self.epsilon = None
    
    def on_train_begin(self, metrics_names, **kwargs):
        return { 'metrics_names': metrics_names + [ self.name ] }
    
    def on_epoch_end(self, last_output, last_target, last_metrics, **kwargs):
        prediction = last_output.detach()
        prediction2 = self.activation(prediction) if self.activation else prediction
        target = last_target.detach()
        target2 = self.epsilon(target) if self.epsilon else target
        result = self.loss_fn(prediction2, target2)
        last_metrics.append(result)
        return { 'last_metrics': last_metrics }



def quad_mean_loss(predicted, target):
    """Take the fourth power of the difference between input and target.
    This is intended as a form of Focal Loss (loss that penalizes being 'very wrong' much more than
    being 'a little wrong').  See https://arxiv.org/abs/1708.02002, in particular Appendix A"""
    return torch.pow( predicted - target, 4).mean()


def other_dimensions(ndims, exclude):
    """Return a tuple of the dimensions up to ndims, _not_ including those in exclude."""
    return list(set(range(ndims)) - set(listify(exclude)))

def class_weights(target, weight_index=1):
    """Return a vector of relative weights, one for each class in target.  In the non-binary case, it is a vector of relative density."""
    # we need a vector of all other indexes except the weight index.
    other_dims = other_dimensions(len(target.shape),weight_index)
    wts = target.sum(other_dims)
    return wts / wts.sum()

# There are a lot of different definitions of IoU and Dice scores, and while some of them are just algebraic rearrangements
# of the same formula, some aren't.  For the record, here is how I am interpreting the original binary measures:
#
#  IoU  = intersection / union 
#       = TP / (TP + FP + FN)
#  Dice = 2 * intersection / (union + intersection) 
#       = (2 * TP) / ((2 * TP) + FP + FN)
#
#  Since IoU and Dice are measures of similarity,  1 - Dice and 1 - IoU are measures of loss

def weighted_dice(predicted, target, weight_index=1):
    """Weighted Dice loss measure, computed separately for each class, then summed.
    The score for each class is weighted by the inverse of it's squared prevalence in the sample.
    See https://arxiv.org/abs/1707.03237."""
    wts = class_weights(target, weight_index)
    wts = 1 / (wts * wts + 0.00001)

    sumdims = other_dimensions(len(target.shape),weight_index)
    intersection = (target * predicted).sum(sumdims)
    intersection = (intersection * wts).sum()

    uplusi = (target + predicted).sum(sumdims)
    uplusi = (uplusi * wts).sum()

    return 2 * intersection / uplusi

def dice(predicted, target):
    intersection = (target * predicted).sum()
    union = (target + predicted).sum()
    return 2 * ( intersection / union )
