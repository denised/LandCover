import requests
import platform
import hashlib
from datetime import datetime
import torch
from fastai.basic_train import LearnerCallback
from fastai.core import defaults, ifnone

columns = [
        # LS = Learner *must* set 
    'train_id',    # ID used to identify a specific experiment.
    'description', # LS -- Arbitrary content provided by Learner
    'started',     # timestamp when experiment started
    'notes',       # space for hand-supplied notes; should *not* be set by code
    'code_marker'  # LS -- meant to be a constant that marks significant changes in code functionality
    'data',        # Identification of data used for this experiment.
    'augmentation',# LS -- Identification of any data treatment / augmentation used
    'base_model',  # train_id of experiment that generated the model weights used to initialize this experiment.
    'arch',        # LS -- Identification of the architecture used
    'loss_func',   # Identification of the Loss Function used
    'optimizer',   # Identification of the optimizer used
    'invocation',  # LS -- Fit, vs fit-one-cycle, etc.
    'epochs',      # LS -- Number of full epochs the experiment is to run for
    'parameters'   # LS -- top-level parameters such as learning rate, weight decay
    'completed',   # total number of batches that were completed
    'metrics',     # Some small set of statistics identifying the loss/accuracy, etc. for the entire run
    'machine',     # Name of machine where this experiment runs
    'error',       # Error information if the experiment fails due to an exception
    'duration'     # How long the experiment took before completion or error
    ]

class TrainTracker(LearnerCallback):
    """TrainTracker keeps track of experiments.  It is kinda similar to Neptune.ml or Comet.ml or Datmo (etc.), but has
    a few different design rationale:
      * Not: deeply understand an experiment via statistics generated during the run (use Neptune, et al. for that)
      * Not: save enough information to reproduce any possible experiment.  (The resources required to do this for 
        data at LandCover scale would be too expensive, in both time and money.)
      * Yes: keep track of enough metadata to enable some analysis across multiple experiments, or enable finding 
        experiment(s) that had certain characteristics.
      * Yes: keep track of lineage of model weights across multiple experiments (when a model is pre-warmed from existing weights)
      * Yes: manage an ID that can be used to cross-correlate data in this Tracker with other kinds of tracking (Neptune, log files, etc.)
    
    and finally:
      * Not: automatically generate all tracked metadata without you having to write any code.  
      * Instead: augment/annotate the classes of fastai (or our extensions of it) to create reasonable metadata with minimal work.
        (But of course that means it only works for fastai, and only when fastai has been appropriately augmented.)
    
    Various mechanisms are used to obtain the data stored.
     * Values are stored in the 'parameters' field of Learner.  TrainTracker has default methods to set most of them, but Learner
       can set them first, or override them afterwards.  A few values must be set by the Learner.
     * Optimizers and Loss Functions are searched for a tracker_description method or attribute.  If found, that is used
       to identify the respective object.  If it is not found, a default mechanism is used to identify Loss Functions or Optimizers.
    *  Adds an extra parameter to models before saving to allow them to store and recover the train_id.

    This code is hardwired into the learner in this repo.  You do not need to manually add it to callbacks.
    """
    _order=-99

    # pylint: disable=arguments-differ
    
    def on_train_begin(self, **kwargs):
        self.started = datetime.now()
        p = self.learn.parameters
        p['machine']    = platform.node()
        p['train_id']   = self.generate_id()
        p['started']    = self.started.strftime('%d/%m/%y %H:%M')
        p['base_model'] = self.get_model_id(self.learn.model)
        if 'loss_func' not in p: p['loss_func'] = self.describe(self.learn.loss_func)
        if 'optimizer' not in p: p['optimizer'] = self.describe(self.learn.opt_func)
        if 'data'      not in p: p['data'] = self.describe_data()
        
        if 'metrics' in p:   del p['metrics']
        if 'error' in p:     del p['error']
        if 'duration' in p:  del p['duration']
        if 'completed' in p: del p['completed']
    
    def on_train_end(self, iteration, **kwargs):
        p = self.learn.parameters
        p['completed'] = iteration
        p['duration'] = str(datetime.now()-self.started)
        p['metrics'] = self.get_metrics(kwargs)
        self.set_model_id(self.learn.model, p['train_id'])

        if defaults.traintracker_store:
            defaults.traintracker_store.emit(p)
    
    def on_error(self, exception):
        p = self.learn.parameters
        p['completed'] = 0  #stackfind('something', 'iteration')
        p['duration'] = datetime.now()-self.started
        p['error'] = exception 
    
    ##################################################################################################################
    # All the ways we find, store, create, this data.
    
    def generate_id(self):
        # we make these fixed width so that we can embed them in a tensor (see set_model_id)
        mname = self.learn.parameters['machine'][-10:]
        return f"{self.started:%Y%m%d.%H%M}.{mname:10.10}"
   
    def describe(self, ob):
        """If the object has a tracker_id attribute, use that to produce the result.  Otherwise simply apply the str() method"""
        if hasattr(ob, 'tracker_id'):
            x = ob.tracker_id
            return str(x() if callable(x) else x)
        else:
            return str(ob)
    
    @classmethod
    def describe_parameters(cls, learner, lr=None, wd=None, **kwargs):   # pylint: disable=unused-argument
        # I don't like that I'm copying the fastai code, but I don't have any way to intercept the state that fit computes internally
        # Also, we should be able to expand with additional parameters as needed.
        lr = ifnone(lr, defaults.lr)
        wd = ifnone(wd, learner.wd)
        bs = learner.data.train_dl.batch_size
        return f"lr={lr} wd={wd} bs={bs}"
    
    # ######## Sneak an id into model data so that we can retrieve it later
    #
    @classmethod
    def set_model_id(cls, model, model_id):
        """Add the id to model, so we can retrieve it again later.  This should be done _after_ a run, but 
        before the model is saved."""
        # The string has to be fixed length to prevent torch from complaining when reloading.
        model_id = f"{model_id:24.24}"
        encoded_id = torch.tensor( list(model_id.encode()), dtype=torch.uint8 )  # pylint: disable=not-callable,no-member
        if not hasattr(model, 'tracker_id'):
            model.register_buffer('tracker_id', encoded_id)
        else:
            model.tracker_id = encoded_id

    @classmethod
    def get_model_id(cls, model):
        """Extract the identifier for the model data, if it is present"""
        if hasattr(model, 'tracker_id'):
            return bytearray(model.tracker_id.tolist()).decode()
        else:
            return "empty"
    
    # ######## Describe the data used for this run
    # Most of the functionality in TrainTracker is at least somewhat generalized or generalizable.
    # This function is the main exception: it knows *exactly* how my data is mangaged --- for example there no place else
    # in the code that knows there is a csv file kept in the same directory as the data files.
    # So someday, need to re-architect to not be so magic.

    def describe_data(self):
        """Return an identifier for the current data.  This implementation is very magic (understands my data setup)."""
        # (1) The lengths of the training and validation sets
        # (2) A hash signature of those sets
        # The hash signature alone would be enough, but the lengths are easier to read quickly and may be useful info as well.
        # Someday we'd like to add the sample-generation method as well
 
        train_set = self.learn.data.train_ds.windows
        val_set = self.learn.data.valid_ds.windows
        
        encoder = hashlib.md5()
        encoder.update(b"train")
        for w in train_set:
            encoder.update( str(w).encode() )
        encoder.update(b"val")
        for w in val_set:
            encoder.update( str(w).encode() )
        
        return f"t:{len(train_set)} v:{len(val_set)} {encoder.hexdigest()}"


    def get_metrics(self, kwargs):
        stats = [ kwargs['last_loss']] + kwargs['last_metrics']
        names = ['tloss', 'vloss'] + self.learn.recorder.metrics_names
        printed = []
        for n,s in zip(names, stats):
            printed.append( n + '=' + (str(s) if isinstance(s, int) else '##' if s is None else f'{s:.6f}'))
        return " ".join(printed)
    
    def __str__(self):
        return "TrainTracker()"


# def stackfind(param_name, stack=None, matching=None):
#     """Find the value of a parameter named param_name on traceback stack (the most recent error traceback by default).
#     If matching is provided, the function name must contain that substring."""
#     # TODO
#     return "0"

#########################################################
# Where the data gets written.  A webhook is one way to do it, but you can use whatever other technique you would like
# by assigning a different object to defaults.traintracker_store
#
# In my setup, I use a Google Sheets document as my target, with a scripted webhook following these instructions:
# http://mashe.hawksey.info/2014/07/google-sheets-as-a-database-insert-with-apps-script-using-postget-methods-with-ajax-example/

class TrainTrackerWebHook(object):
    def __init__(self, service_uri):
        self.service_uri = service_uri
    
    def emit(self, values):
        """values is a dictionary of field values to write.  Writing is append-only, so all the data about a single training
        run needs to be written in a single call."""
        
        return requests.post(self.service_uri, values).json()   # return value is just for debugging.
