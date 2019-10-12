# Sample code to set up and do a run by script
import argparse
import datetime
from fastai import *
from fastai.basics import *
from multispectral import corine
from multispectral import windows
import neptune
import infra
import zoo


# If working interactively, you probably also want
# import numpy as np
# import torch
# from matplotlib import pyplot
# from multispectral import tools
# import imp

parser = argparse.ArgumentParser()
parser.add_argument("--shutdown", help="shutdown computer when complete", action='store_true')
parser.add_argument("--description", help="description of run", default="runner.py")
parser.add_argument("--epochs", type=int, help="how many full epochs to run (default 1)", default=1)
parser.add_argument("--cpu", help="run on cpu", action='store_true')
args = parser.parse_args()

corine.set_corine_directory(os.environ['CORINE_DIR'])
windows_list = os.environ['LANDSAT_DIR'] + '/all_windows.csv'
neptune.init('denised/landcover')
infra.set_defaults(traintracker_store=infra.TrainTrackerWebHook(os.environ['TRAINTRACKER_URI']))

wl = list(windows.from_file(windows_list))
(tr_list, val_list) = windows.chunked_split(wl,512)
#tr_list = tr_list[:10000]

#ender = infra.TrainEnder()
infra.set_defaults(bs=4,silent=True,train_end=4)
if args.cpu:
    infra.set_defaults(device=torch.device('cpu'))
logfilename = "runnerlog_{:%y%m%d_%H%M%S}.csv".format(datetime.datetime.now())


def run_one():
    monitor=infra.CycleHandler.create(n=2, callbacks=[
        infra.DiceMetric(), infra.GradientMetrics, infra.LearnedClassesMetric(),
        infra.CSVLogger(logfilename,'a'), infra.SendToNeptune])
         
    learner = zoo.MultiUResNet.create(tr_list, val_list, callbacks=[], callback_fns=[monitor])
    learner.fit(args.epochs, description=args.description)   # pylint: disable=unexpected-keyword-arg
    
    # modelname="foo.pkl"
    # torch.save(learner.model.state_dict(), modelname)
    
    
try:
    run_one()
finally:
    if args.shutdown:
        os.system("sudo shutdown now")
        
    

