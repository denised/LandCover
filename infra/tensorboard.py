from torch.utils.tensorboard import SummaryWriter
from fastai.torch_core import ModelOnCPU
from fastai.callbacks import Callback
from pathlib import Path
import re

class TensorboardLogger(Callback):
    """ """

    def __init__(self, param_regex='.*', gradient_regex=None, add_graph=True, add_metrics=True, learner=None):
        """ """
        super().__init__()
        self.param_regex=param_regex
        self.gradient_regex=gradient_regex
        self.do_graph=add_graph
        self.do_metrics=add_metrics
        self.do_gradients = False
        self.learn = learner
        self.tb = None
    
    def setLearner(self, learner):
        self.learn = learner

    def on_train_begin(self, **kwargs):
        self.metric_names = ['loss'] + kwargs['metrics_names']

        # Override directory name if we have a train_id
        train_id = self.learn.parameters.get('train_id', None)
        log_dir = str(Path( 'runs' ) / train_id.rstrip()) if train_id else None
        self.tb = SummaryWriter( log_dir=log_dir )

        # Write some of the hyperparameters
        params = self.learn.parameters
        # pass a subset of hyperparameters to Tensorboard
        if 'arch' in params:
            self.tb.add_text('arch',params['arch'])
        if 'loss_func' in params:
            self.tb.add_text('loss_func', params['loss_func'])
        if 'parameters' in params:
            self.tb.add_text('hyperparameters', params['parameters'])

    def on_batch_end(self, **kwargs):
        if self.do_graph:
            with ModelOnCPU(self.learn.model) as m:
                inp = kwargs['last_input'].detach().cpu()
                self.tb.add_graph(m, inp)
            self.do_graph=False

    def on_epoch_begin(self, **kwargs):
        # doing this in epoch_begin instead of epoch_end allows us to have matching weights and gradients
        if self.param_regex:
            batchno = kwargs['num_batch']
            for n, p in self.learn.model.named_parameters():
                if re.match(self.param_regex, n):
                    self.tb.add_histogram(n,p.data,batchno)
        self.do_gradients = True
    
    def on_backward_end(self, **kwargs):
        if self.do_gradients and self.gradient_regex:
            batchno = kwargs['num_batch']
            for n, p in self.learn.model.named_parameters():
                if re.match(self.gradient_regex,n) and p.grad is not None:
                    self.tb.add_histogram(n+'_grad',p.grad,batchno)
            self.do_gradients = False
    
    def on_epoch_end(self, **kwargs):
        if self.do_metrics:
            batchno = kwargs['num_batch']
            stats = [kwargs['last_loss']] + kwargs['last_metrics']
            for name, val in zip(self.metric_names, stats):
                self.tb.add_scalar(name, val, batchno)

    def on_train_end(self, **kwargs):
        self.tb.close()
