from contextlib import contextmanager
from pathlib import Path
from functools import partial
from matplotlib import pyplot
import torch
import fastai
from fastai.basic_train import * #Learner
from fastai.callbacks import * #Callback
from fastai.basic_data import DataBunch, DeviceDataLoader
from torch.utils.data.dataloader import DataLoader
from fastai.core import defaults, ifnone
from fastai.torch_core import rank_distrib, try_save

################################################
# Our augmentation of the fastai Learner class (and related classes)
# * Merge the creation of Learner and DataBunch
# * Make it easy to substitute new data into an existing Learner
# * Support export/import of Learner state without Dataset
# * Do end-epoch stuff every n batches instead of the actual epoch length

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
    def temporary_validation_set(self, x_dataset, pause_train=True):
        """Override the validation set for the duration of this context.  This is usually done to perform evaluation of a specific set of data.
        If pause_train is true, the model is also set to eval for the duration (and restored to its previous setting at the end)."""
        stashed_train_state = self.model.training
        stashed_validation_loader = self.data.valid_dl
        self.data.valid_dl = DeviceDataLoader(x_dataset.as_loader(), self.data.device)
        if pause_train: self.model.eval()
        yield True
        self.data.valid_dl = stashed_validation_loader
        if pause_train: self.model.train(stashed_train_state)
    
    @contextmanager
    def pause_training(self):
        """Set model to eval temporarily.  Model is restored to initial state afterwards (which might have been either train or eval)."""
        stashed_train_state = self.model.training
        self.model.eval()
        yield True
        self.model.train(stashed_train_state)
    
    @classmethod
    def _init_args(cls, opt_func=None, loss_func=None, metrics=None, true_wd=True, bn_wd=True, wd=None, train_bn=True,
                   path=None, model_dir=None, callback_fns=None, callbacks=None, layer_groups=None, add_time=True, silent=None, 
                   **kwargs):
        """Pull out arguments for the learner class and set their defaults.  Returns a tuple learner-args, other-args"""
        # Why we do this seemingly pointless thing:  this allows our Learner create methods to accept a mixed kwargs that covers both Learner and DataBunch
        # Here we (a) separate the mixed kwargs into two lists, and (b) standardize the default values for the learner args (which include additional
        # defaults beyond what fastai does)
        learner_args = {
            'opt_func': ifnone(opt_func, defaults.opt_func),
            'loss_func': ifnone(loss_func, defaults.loss_func),
            'metrics': ifnone(metrics, defaults.metrics),
            'true_wd': true_wd,
            'bn_wd': bn_wd,
            'wd': ifnone(wd, defaults.wd),
            'train_bn': train_bn,
            'path': ifnone(path, defaults.model_directory),
            'model_dir': ifnone(model_dir, '.'),  # This is a change from fastai default: we mix leaner exports and models in same dir
            'callback_fns': ifnone(callback_fns, defaults.callback_fns),
            'callbacks': callbacks,
            'layer_groups': layer_groups,
            'add_time': add_time,
            'silent': silent
        }
        return learner_args, kwargs
    
    # This is crappy architecture, but it will work for now.
    def _extra_run_params(self, kwargs):
        """Add a few extra keyword parameters that we will accept in fit, lr_find, etc."""
        # If a 'description' keyword is present, store it (handed off to neptune)
        if 'description' in kwargs:
            self.description = kwargs['description']
            del kwargs['description']
        
        # If a 'neptune' keyword is present, that will control whether neptune records this run or not
        if 'neptune' in kwargs:
            self.do_neptune = kwargs['neptune']
            del kwargs['neptune']
        
        # if 'epoch_length' in kwargs:
        #     self.epoch_length = kwargs['epoch_length']
        #     del kwargs['epoch_length']
        # else:
        #     self.epoch_length = defaults.epoch_length

    def fit(self, *args, **kwargs):
        self._extra_run_params(kwargs)
        #import pdb; pdb.set_trace()
        # if self.epoch_length:
        #     # add callback shortener to callbacks list if requested
        #     callbacks = kwargs['callbacks'] if 'callbacks' in kwargs else []
        #     callbacks.append( EpochShortener(self.epoch_length) )
        #     kwargs['callbacks'] = callbacks
        super().fit(*args, **kwargs)
        # clean up; only needed here in fit b/c the fit_one_cycle and lr_find both call fit
        if hasattr(self,'do_neptune'): del self.do_neptune
    
    def fit_one_cycle(self, *args, **kwargs):
        self._extra_run_params(kwargs)
        super().fit_one_cycle(*args, **kwargs)
    
    def lr_find(self, *args, **kwargs):
        self._extra_run_params(kwargs)
        super().lr_find(*args, **kwargs)

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

    
class EpochCleaner(LearnerCallback):
    _order = -10
    """Clean up some stuff (currently just the per-batch recorder data) every epoch"""
    def on_epoch_begin(self, **kwargs):
        self.learn.recorder.losses = []
        self.learn.recorder.lrs = []
        self.learn.recorder.moms = []


class CycleHandler(Callback):
    """The cycle handler manages a set of other callbacks on a cycle of length n.
    That is, to the callbacks and metrics that are being managed, it appears that the epoch length is n, all the while the top-level CallbackHandler
    is going through the true epoch.  As much as possible, callbacks can be used interchangeably between CallbackHandler and CycleHandler,
    but there are some things to keep in mind:
    * Callbacks or metrics passed in directly to calls to fit, etc., go into the main top-level CycleHandler.  To get one or more callbacks to work 
    on an n-cycle basis, you must explicitly create a CycleHandler for it/them.
    * If a CycleHandler contains metrics, it should also contain one or more callbacks that do something with the metric data (such as CSVLogger),
    since the default CycleHandler behavior of passing the data to the Recorder does not happen.
    * Any other state manipulation (such as modifying the learning rate, or stopping training) does "go through" to the main CallbackHandler.
    * Validation is not done automatically at the end of a cycle, but it can be done if desired by adding the Validate callback.
    * CycleHandler treats callbacks and metrics the same, and the order in which they are supplied is the order in which they will be called.
    * Starting training at a particular epoch is not supported (since the outer callback handler and the cycles don't have a common definition of what
    an epoch _is_)
    * Having multiple CycleHandler's (including with different values of n) is possible; they will operate completely independently of each other."""

    @classmethod
    def create(cls, n, callbacks):
        """Partially initialize a CycleHandler.  Returns a function that will create the actual callback on demand, allowing late binding of
        the learner.  Use this by passing the result to the `callback_fns` argument of Learner like this:
            ch = CycleHandler.create(200, [__list of callbacks__])
            mylearner = Learner(..., callback_fns=[ch])"""
        return partial(cls, n=n, callbacks=callbacks)

    def __init__(self, learner, n, callbacks):
        """Initialize the cyclehandler for a cycle length of n batches.  Callbacks must be an array, each element of which may be one
        of following three things:
            * a callback object
            * a class (which is instantiated to create a callback object), or
            * a function.  Functions are assumed to compute a metric over a batch, and are automatically wrapped in AverageMetric to return the average of
            that function over all batches in the cycle.
        Collectively this covers the cases supported via various features of Learner and CallbackHandler"""
        super().__init__()
        self.n = n
        self.count = 0
        self.callbacks = []
        for c in callbacks:
            if isinstance(c, type):
                c = c(learner)
            elif isinstance(c, Callable):
                c = AverageMetric(c)
            self.callbacks.append(c)
    
    def _propagate(self, event_name, **cbstate):
        """Propagate the named event to all our managed callbacks, and collect the responses"""
        # Why does CallbackHandler overload __call__ to do this?  So much clearer as a named function
        delta = {}
        for c in self.callbacks:
            result = getattr(c, event_name)(**cbstate)  # call the appropriate event
            if result:
                cbstate.update(result)
                delta.update(result)
        return delta

    def _return(self, delta):
        """Clean up the result we return from an event"""
        # Everywhere in fastai, both stop_training and stop_epoch are set to true at once.
        # Without a real use case, the right behavior isn't clear to me, so just leaving it be...
        # if 'stop_epoch' in delta: 
        #     delta['stop_training'] = delta['stop_epoch']
        # Anything having to do with metrics is local to this cycle, so don't send it back up the foodchain
        if 'last_metrics' in delta: 
            del delta['last_metrics']
        if 'metrics_names' in delta:
            del delta['metrics_names']
        return delta
    
    # Most of the rest of this is very boring.  The interesting bits are called out with  # ***

    def on_train_begin(self, **cbstate):
        #import pdb; pdb.set_trace();
        self.count = 0;
        return self._return(self._propagate('on_train_begin', **cbstate))
    
    def on_epoch_begin(self, **cbstate):
        #log_it(self, 'on_epoch_begin', **cbstate)
        pass                                    # *** handled below, in on_batch_begin
    
    def on_batch_begin(self, **cbstate):        # ***
        # What's going on here: we begin an epoch if we are on the cycle boundary and we always begin a batch.
        # We need to accumulate delta from both operations, and we need to update cbstate in between them.
        #log_it(self, 'on_batch_begin', **cbstate)
        delta = {}
        if self.count % self.n == 0:
            delta.update( self._propagate('on_epoch_begin', **cbstate) ) 
            cbstate.update(delta)
        delta.update( self._propagate('on_batch_begin', **cbstate) )
        self.count += 1
        return self._return(delta)
    
    def on_loss_begin(self, **cbstate):
        return self._return(self._propagate('on_loss_begin', **cbstate))
    
    def on_backward_begin(self, **cbstate):
        return self._return(self._propagate('on_backward_begin', **cbstate))
    
    def on_backward_end(self, **cbstate):
        return self._return(self._propagate('on_backward_end', **cbstate))
    
    def on_step_end(self, **cbstate):
        return self._return(self._propagate('on_step_end', **cbstate))
    
    def on_batch_end(self, **cbstate):          # ***
        #log_it(self, 'on_batch_end', **cbstate)
        # Always do a batch_end, and if this is the end of a cycle also do an epoch_end
        delta = self._propagate('on_batch_end', **cbstate)
        if self.count % self.n == 0:
            cbstate.update(delta)
            cbstate['last_metrics'] = []
            delta.update( self._propagate('on_epoch_end', **cbstate) )
        return self._return(delta)
    
    def on_epoch_end(self, **cbstate):          # ***
        #log_it(self, 'on_epoch_end', **cbstate)
        pass     # handled above.  Note we will miss the last partial cycle of the last epoch if it doesn't divide evenly into n
          # However, it only affects the last epoch since cycles can span across epoch boundaries.

    def on_train_end(self, **cbstate):
        return self._return(self._propagate('on_train_end', **cbstate))

    def jump_to_epoch(self, epoch):
        # Not supported
        pass
    
    # TODO: do we need to override __repr__ or is it okay?


class Validate(LearnerCallback):
    def __init__(self, learn, callbacks=None):
        """Run validation on demand, potentially with a list of callbacks.  The callbacks would most likely be used to do test time augmentation (TTA)."""
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

class LearnTracer(fastai.basic_train.LearnerCallback):
    """Print out various things as they occur.  Best used with "silent=True" argument to fit.
    Checks for nans on forward and backward, and also checks for zero-grads on backward"""

    def on_train_begin(self, pbar, **kwargs):
        self.pbar = pbar
        self.hooks = []
        
        def hook(module, input, output, name):
            if module.training and torch.isnan(output).sum().item() > 0:
                print("warning: {} output: nan(s)".format(name))
                if defaults.trace_pdb:
                    import pdb; pdb.set_trace()

        for n, m in self.learn.model.named_modules():
            self.hooks.append( m.register_forward_hook( partial(hook, name=n) ))

    def on_batch_end(self, num_batch, last_loss, **kwargs):
        if num_batch > 0 and num_batch % defaults.trace_frequency == 0:
            with self.learn.pause_training():
                result = validate(self.learn.model, self.learn.data.valid_dl, loss_func=self.learn.loss_func, pbar=self.pbar)
                print("loss: {:8.4f}  val_loss: {:8.4f}".format( last_loss, result ))
        else:
            print(".", end='')
    
    def on_backward_end(self, num_batch, **kwargs):
        # periodically report the gradient sum range and mean
        if num_batch > 0 and num_batch % defaults.trace_frequency == 0:
            stats = np.array([ p.grad.abs().sum().item() for p in self.learn.model.parameters() if p.grad is not None])
            if len(stats) > 0:
                print("batch {:3d}: grad mags: min {:8.3f}, max {:8.3f}, mean {:8.3f}; ".format( num_batch, stats.min(), stats.max(), stats.mean()), end='' )
            else:
                print("No grads found", end='')
        
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

class TrainEnd(fastai.basic_train.LearnerCallback):
    # Note: fastai has a callback just like this, but I like setting the trace_end parameter via defaults instead of __init__.
    # Makes initializing it cleaner.
    def on_batch_end(self, num_batch, **kwargs):
        if defaults.train_end and num_batch >= defaults.train_end:
            return {'stop_epoch': True, 'stop_training': True, 'skip_validate': True}


class LRAccumulator(object):
    """Accumulate multiple recorder results to compare them on the same graph.  Can be applied across any Learner fit method
    (lr_find, fit, etc.), and a single accumulator can be used across multiple learners, models, data... anything where you'd like
    to compare the loss graphs."""
    def __init__(self, learner=None, title="a", fmt=''):
        """Create a new accumulator, optionally starting with an initial recorder trace."""
        self.curves = []
        if learner:
            self.add(learner, title, fmt)
 
    def add(self, learner, title=None, fmt=''):
        """Add another recorder trace to the list.
        The format of the curve can be specified with the fmt argument using the matplotlib format shortcut notation (e.g. 'ro-')"""
        title = ifnone(title, chr(ord("a") + len(self.curves)))
        self.curves.append( (title, learner.recorder.lrs, [x.item() for x in learner.recorder.losses], fmt) )
    
    def drop(self, index=-1):
        """Add the wrong curve by mistake?"""
        del self.curves[index]
    
    def plot(self, bylrs=True, xmin=None,xmax=None,ymin=None,ymax=None):
        """Plot all the accumulated curves.  By default, plots loss against learning rate (which is appropriate for comparing lr_find
        results).  To compare other loss traces, set `bylrs=False`.  By default the graph will be scaled to include all the data for
        all the curves; use the xmin/max and ymin/max arguments to focus on the interesting part."""
        _, ax = pyplot.subplots(1,1)
        for (label, xs, ys, fmt) in self.curves:
            if bylrs:
                ax.plot(xs, ys, fmt, label=label)
            else:
                ax.plot(ys, fmt, label=label)
        ax.set_ylabel("Loss")
        ax.set_xlabel("Learning Rate" if bylrs else "Batch")
        if xmin is not None or xmax is not None:
            ax.set_xlim(left=xmin, right=xmax)
        if ymin is not None or ymax is not None:
            ax.set_ylim(bottom=ymin, top=ymax)
        if bylrs: 
            ax.set_xscale('log')
            ax.xaxis.set_major_formatter(pyplot.FormatStrFormatter('%.0e'))
        ax.legend()
