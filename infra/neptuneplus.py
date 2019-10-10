import logging
import fastai
import neptune

###############################################
# Neptune integration
#
# Automatically set up netpune experiments and log training parameters on each call to
# learner.fit, using a fastai callback mechanism.  
# 
# Note: this is equivalent to/ replacement for neptune_contrib NeptuneMonitor, which does a couple
# of things differently:
#  * recognizes if neptune hasn't been initialized and continues anyway
#  * auto initialization of experiments (tied to training runs)
#  * knows about and uses our traintracker parameters

_logger = logging.getLogger(__name__)

class SendToNeptune(fastai.basic_train.LearnerCallback):
    def __init__(self, learner):
        super().__init__(learner)
        self.exp = None
        self.own_exp = False  # if true, we created the experiment, so we should stop it

    def on_train_begin(self, metrics_names, **kwargs):        # pylint: disable=arguments-differ
        # check to see if the user has turned neptune off
        if getattr(self.learn, 'do_neptune', True) == False:
            return

        try:  # if the caller has already created an experiment, use that instead.
            self.exp = neptune.get_experiment()
        except neptune.exceptions.NoExperimentContext:
            # we normally expect to end up here.
            # get the parameters of this training to pass to neptune
            params = getattr(self.learn, 'parameters', {})
            title = getattr(self.learn, 'title', '')
            description = getattr(self.learn, 'description', '')
            try:
                self.exp = neptune.create_experiment(name=title, description=description, params=params)
                self.own_exp = True
            except neptune.exceptions.Uninitialized:
                _logger.warn("Neptune not initialized; no tracing will be done")
        
        # This would not work in regular fastai, because metrics_names is not updated with additional names
        # But in our case, we use this inside a CycleHandler, which does update them.
        self.metrics_names = ['train_loss'] + metrics_names

    def on_train_end(self, **kwargs):
        if self.exp and self.own_exp:
            self.exp.stop()
        self.exp = None
        self.own_exp = False

    def on_epoch_end(self, **kwargs):
        if self.exp:
            # get the training loss, validation loss, and whatever other metrics we have recorded
            # fastai puts the validation loss as the first item in last_metrics
            x = kwargs['iteration']
            metric_vals = [kwargs['last_loss']] + kwargs['last_metrics']

            for metric_name, metric_value in zip(self.metrics_names, metric_vals):
                if metric_value is not None:
                    self.exp.send_metric(metric_name, x, float(metric_value))


