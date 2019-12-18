from functools import partial
from fastai.callback import Callback, AverageMetric
from fastai.basic_train import LearnerCallback

class CycleHandler(LearnerCallback):
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
    * Having multiple CycleHandler's (including with different values of n) is possible; they will operate completely independently of each other.
    * CallbackHandler by default only operates during training; set the on_eval argument to True to have it operate during the main-level validation
      as well."""

    @classmethod
    def create(cls, n, callbacks, on_eval=False):
        """Partially initialize a CycleHandler.  Returns a function that will create the actual callback on demand, allowing late binding of
        the learner.  Use this by passing the result to the `callback_fns` argument of Learner like this:
            ch = CycleHandler.create(200, [__list of callbacks__])
            mylearner = Learner(..., callback_fns=[ch])"""
        return partial(cls, n=n, callbacks=callbacks, on_eval=on_eval)

    def __init__(self, learner, n, callbacks, on_eval=False):
        """Initialize the cyclehandler for a cycle length of n batches.  Callbacks must be an array, each element of which may be one
        of following three things:
            * a callback object
            * a class which can be instantiated to create a callback object, or
            * a function.  Functions are assumed to compute a metric over a batch, and are automatically wrapped in AverageMetric to return the average of
            that function over all batches in the cycle.
        Collectively this covers the cases supported via various features of Learner and CallbackHandler"""
        super().__init__(learner)
        self.n = n
        self.on_eval = on_eval
        self.count = 0
        self.callbacks = []
        for c in callbacks:
            if isinstance(c, Callback):
                if hasattr(c, 'setLearner'):
                    c.setLearner(learner)
            elif isinstance(c, type):
                c = c(learner)
            else: 
                assert callable(c)
                c = AverageMetric(c)
            self.callbacks.append(c)
    
    def _propagate(self, event_name, **cbstate):
        """Propagate the named event to all our managed callbacks, and collect the responses"""
        # Why does CallbackHandler overload __call__ to do this?  So much clearer as a named function
        delta = {}
        cbstate['cyclehandler'] = True  # make it possible for the callback to tell if it is within cyclehandler or not
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
        if cbstate['train'] or self.on_eval:
            delta = {}
            if self.count % self.n == 0:
                delta.update( self._propagate('on_epoch_begin', **cbstate) ) 
                cbstate.update(delta)
            delta.update( self._propagate('on_batch_begin', **cbstate) )
            self.count += 1
            return self._return(delta)
    
    def on_loss_begin(self, **cbstate):
        if cbstate['train'] or self.on_eval:
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
        if cbstate['train'] or self.on_eval:
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

