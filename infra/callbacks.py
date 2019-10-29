import torch
from fastai.basic_train import *
from fastai.core import defaults, ifnone, listify
from fastai.callback import Callback
from functools import partial
from pathlib import Path

# pylint: disable=arguments-differ

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
        
        def hook(module, input, output, name):  # pylint: disable=unused-argument,redefined-builtin
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

class TrainEnder(Callback):
    # Note: fastai has a callback just like this, but I like setting the train_end parameter via defaults instead of __init__.
    # Makes initializing it cleaner.
    def on_batch_end(self, num_batch, **kwargs):
        if defaults.train_end and num_batch >= defaults.train_end:
            return {'stop_epoch': True, 'stop_training': True, 'skip_validate': True}


class CSVLogger(Callback):
    """Like the fastai version, but reads metrics_names from the callback handler, instead of the recorder"""
    def __init__(self, filename='history.csv', mode='w'):
        super().__init__()
        self.path = Path(filename)
        self.mode = mode
        self.file = None
    
    def on_train_begin(self, metrics_names, **kwargs):
        "Prepare file with metric names."
        names = ['batchno', 'loss'] + metrics_names
        self.path.parent.mkdir(parents=True, exist_ok=True)      
        self.file = self.path.open(self.mode)
        self.file.write(','.join(names) + '\n')
    
    def on_epoch_end(self, iteration, last_loss, last_metrics, **kwargs):
        stats = [str(stat) if isinstance(stat, int) else '##' if stat is None else f'{stat:.6f}'
                 for stat in [iteration, last_loss]+last_metrics ]
        self.file.write(','.join(stats) + '\n')
    
    def on_train_end(self, **kwargs):
        self.file.close()
        self.file=None

#############################################################################################################
#
# Loss and Metric Wrapper / Tools
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
    """Wrapper around a loss function to apply activation and/or epsilon shrinking of target.  Use as loss_func in training."""
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
        target2 = self.epsilon(target) if self.epsilon else target.float()
        return self.loss_fn(prediction2, target2)


class LossMetric(Callback):
    """Wrapper around a function to apply activation if needed.  Use for a metric/callback.
    LossMetric is applied to the last batch of an epoch only, so is suitable for sampling overall trends,
    but not for evaluating entire history.  Use a custom callback or AverageMetric for other behaviors.
    """
    _order = -10

    def __init__(self, fn, name, activation=torch.sigmoid):
        super().__init__()
        self.loss_fn = fn
        self.name = name
        self.activation = activation
    
    def on_train_begin(self, **kwargs):
        if 'cyclehandler' in kwargs:
            return { 'metrics_names': kwargs.get('metrics_names',{}) + [ self.name ] }
    
    def on_epoch_end(self, last_output, last_target, **kwargs):
        prediction = last_output.detach()
        prediction2 = self.activation(prediction) if self.activation else prediction
        result = self.loss_fn(prediction2, last_target)
        return { 'last_metrics': kwargs.get('last_metrics',{}) + [ result ] }
    
    def __str__(self):
        return self.name


#############################################################################################################
#
# Metrics
#
#############################################################################################################

class GradientMetrics(LearnerCallback):
    """Report gradient magnitude means.  Gives a very rough estimate of how much the model is changing.
    Gradients are sampled once per epoch (not accumulated over the epoch)."""
    _order = -10
    def on_train_begin(self, **kwargs):
        self.sample = None
        self.names =  ["min GMM", "mean GMM", "max GMM"]
        if 'cyclehandler' in kwargs:
            return { 'metrics_names' : kwargs.get('metrics_names', {}) +  self.names }
        else:
            self.learn.recorder.add_metric_names(self.names)
    
    def on_backward_end(self, **kwargs):
        if self.sample is None:
            gmms = torch.stack([ p.grad.abs().mean() for p in self.learn.model.parameters() if p.grad is not None ])
            self.sample = [ gmms.min().item(), gmms.mean().item(), gmms.max().item() ]
    
    def on_epoch_end(self, **kwargs):
        # it's possible that epoch end get's called before a sample has happened (?)
        last_metrics =  kwargs.get('last_metrics', {}) + (self.sample or [0.0] * 3)
        self.sample = None
        return { 'last_metrics': last_metrics }


class LearnedClassesMetric(LearnerCallback):
    "Report the mean proportions of classes learned."
    _order = -10
    def __init__(self, learn, classnames = None, activation=torch.sigmoid):
        super().__init__(learn)
        self.classnames = classnames or defaults.classnames
        self.activation = activation

    def on_train_begin(self, **kwargs):
        if 'cyclehandler' in kwargs:
            return { 'metrics_names' : kwargs.get('metrics_names',{}) + self.classnames }
        else:
            self.learn.recorder.add_metric(self.classnames)
    
    def on_epoch_begin(self, **kwargs):
        self.stats = []
    
    def on_batch_end(self, last_output, **kwargs):
        last_output = last_output.detach()
        if self.activation: last_output = self.activation(last_output)
        self.stats.append( class_weights(last_output) )
    
    def on_epoch_end(self, **kwargs):
        cumstats = torch.stack(self.stats).mean((0,))
        return { 'last_metrics': kwargs.get('last_metrics',{}) + cumstats.tolist() }


class DiceMetric(LearnerCallback):
    """Report the dice index.  (larger is better)"""
    _order = -10
    def __init__(self, learn, activation=torch.sigmoid, weighted=False):
        self.activation = activation
        self.weighted = weighted
        self.name = 'wdice' if self.weighted else 'dice'
    
    def on_train_begin(self, **kwargs):
        if 'cyclehandler' in kwargs:
            return { 'metrics_names' : kwargs.get('metrics_names', {}) + [ self.name ]}
    
    def on_epoch_begin(self, **kwargs):
        self.samples = []
    
    def on_batch_end(self, last_output, last_target, **kwargs):
        last_output = last_output.detach()
        if self.activation: last_output = self.activation(last_output)
        if self.weighted:
            self.samples.append( weighted_dice(last_output, last_target) )
        else:
            self.samples.append( dice(last_output, last_target))
    
    def on_epoch_end(self, **kwargs):
        return { 'last_metrics' : kwargs.get('last_metrics', {}) + [ torch.stack(self.samples).mean().item() ]}

#############################################################################################################
#
# LOSS Functions
#
#############################################################################################################

def quad_mean_loss(predicted, target):
    """Take the fourth power of the difference between input and target.
    This is intended as a form of Focal Loss (loss that penalizes being 'very wrong' much more than
    being 'a little wrong').  See https://arxiv.org/abs/1708.02002, in particular Appendix A.
    Note: this doesn't seem to work very well (very unstable)."""
    return torch.pow( predicted - target, 4).mean()


# There are a lot of different definitions of IoU and Dice scores, and while some of them are just algebraic rearrangements
# of the same formulae, some aren't.  For the record, here is how I am interpreting the original binary measures:
#
#  IoU  = intersection / union 
#       = TP / (TP + FP + FN)
#  Dice = 2 * intersection / (union + intersection) 
#       = (2 * TP) / ((2 * TP) + FP + FN)
#
# IoU and Dice are measures of similarity,  (1-Dice) and (1-IoU) are measures of loss
#
# To extend IoU and Dice to multi-class, we do ... nothing.  Just applying the same formulae is 
# equivalent to computing intersection and union (cf TP, FP, etc.) per class, then summing over the 
# classes.  When we want to add weights (to compensate for class imbalance in the data), then we 
# explicitly pull out the summation over the class index for that purpose.
#
# All of this is still true even in the case that our target data is both multi-label and non-binary 
# (i.e. a single pixel can be 0.8 Water *and* 0.5 Forest).  In this cases the summations are computing 
# densities (roughly speaking) instead of counting individuals.

def dice(predicted, target):
    # Note: we aren't protecting against div by zero because in the multi-class case at least one class always applies!
    intersection = (target * predicted).sum()
    uplusi = (target + predicted).sum()
    return 2 * intersection / uplusi 

def other_dimensions(ndims, exclude):
    """Return a tuple of the dimensions up to ndims, _not_ including those in exclude."""
    return list(set(range(ndims)) - set(listify(exclude)))

def class_weights(t):
    """Return a vector of normalized weights, one for each class in t."""
    other_dims = other_dimensions(len(t.shape), defaults.class_index)
    wts = t.sum(other_dims)
    return wts / wts.sum()

def weighted_dice(predicted, target):
    """Weighted Dice loss measure, computed separately for each class, then summed.
    The score for each class is weighted by the inverse of it's squared prevalence in the sample.
    See https://arxiv.org/abs/1707.03237."""
    wts = class_weights(target)
    wts = 1 / (wts * wts + 0.00001)

    sumdims = other_dimensions(len(target.shape), defaults.class_index)
    intersection = (target * predicted).sum(sumdims)
    intersection = (intersection * wts).sum()

    uplusi = (target + predicted).sum(sumdims)
    uplusi = (uplusi * wts).sum()

    return 2 * intersection / uplusi
