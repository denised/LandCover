import torch
import numpy as np
from fastai.basic_train import *
from fastai.core import defaults, ifnone
from functools import partial

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
    """Print out various things as they occur.  Best used with "silent=True" argument to fit.
    Checks for nans on forward and backward, and also checks for zero-grads on backward"""

    def on_train_begin(self, pbar, **kwargs):
        self.pbar = pbar
        self.hooks = []
        self.halfmessage = ""
        
        def hook(module, input, output, name):
            if module.training and torch.isnan(output).sum().item() > 0:
                print("warning: {} output: nan(s)".format(name))
                if defaults.trace_pdb:
                    import pdb; pdb.set_trace()

        for n, m in self.learn.model.named_modules():
            self.hooks.append( m.register_forward_hook( partial(hook, name=n) ))

    def on_batch_end(self, num_batch, last_loss, **kwargs):
        if num_batch % defaults.trace_frequency == 0:
            with self.learn.pause_training():
                result = validate(self.learn.model, self.learn.data.valid_dl, loss_func=self.learn.loss_func, pbar=self.pbar)
                print("batch {:3d}: {} loss: {:8.4f}  val_loss: {:8.4f}".format( num_batch, self.halfmessage, last_loss, result ))
        else:
            print(".", end='')
    
    def on_backward_end(self, num_batch, **kwargs):
        # periodically report the gradient sum range and mean
        if num_batch % defaults.trace_frequency == 0:
            stats = np.array([ p.grad.abs().sum().item() for p in self.learn.model.parameters() if p.grad is not None])
            # stash the print messag in a string so that it gets picked up on_batch_end
            # this prevents it from being overwritten by the recorder's status bar on plain terminal
            if len(stats) > 0:
                self.halfmessage = "grad mags: min {:8.3f}, max {:8.3f}, mean {:8.3f}; ".format( stats.min(), stats.max(), stats.mean())
            else:
                self.halfmessage = ""
        
        # always check all the grads for either nan or all-zeros
        # Getting data back from the GPU is expensive, so we first do a check of all grads together, and only if we find an issue
        # do we check them individually.
        
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


class TrainEnder(LearnerCallback):
    # Note: fastai has a callback just like this, but I like setting the trace_end parameter via defaults instead of __init__.
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
