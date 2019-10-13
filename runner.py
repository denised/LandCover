# Sample code to set up and do a run by script
import argparse
import datetime
import platform
from fastai import *
from fastai.basics import *
from multispectral import corine
from multispectral import windows
import neptune
import infra
import zoo

parser = argparse.ArgumentParser()
parser.add_argument("--shutdown", help="Shutdown computer when complete", action='store_true')
parser.add_argument("--description", help="Description of run", default="runner.py")
parser.add_argument("--epochs", type=int, help="How many full epochs to run (default 1)", default=1)
parser.add_argument("--save", type=bool, help="Save resulting model(s)", action='store_true' )
parser.add_argument("--cpu", help="Run on cpu", action='store_true')
args = parser.parse_args()

corine.set_corine_directory(os.environ['CORINE_DIR'])
windows_list = os.environ['LANDSAT_DIR'] + '/all_windows.csv'
neptune.init('denised/landcover')
infra.set_defaults(traintracker_store=infra.TrainTrackerWebHook(os.environ['TRAINTRACKER_URI']))

wl = list(windows.from_file(windows_list))
(tr_list, val_list) = windows.chunked_split(wl,512)
#tr_list = tr_list[:10000]

#ender = infra.TrainEnder()
#infra.set_defaults(train_end=4)
infra.set_defaults(bs=4,silent=True)
if args.cpu:
    infra.set_defaults(device=torch.device('cpu'))

# TODO: I'd like to put a tracker id in here somewhere but it isn't exposed until after the callback has already been created
logfilename = "runnerlog_{:%y%m%d_%H%M%S}.csv".format(datetime.datetime.now())

def run_one():
    monitor=infra.CycleHandler.create(n=80, callbacks=[
        infra.DiceMetric(), infra.GradientMetrics, infra.LearnedClassesMetric(),
        infra.CSVLogger(logfilename,'a'), infra.SendToNeptune])
         
    learner = zoo.MultiUResNet.create(tr_list, val_list, callbacks=[], callback_fns=[monitor])
    learner.fit(args.epochs, description=args.description)   # pylint: disable=unexpected-keyword-arg
    
    if args.save:
        name = Path(infra.defaults.model_directory) / (learner.parameters["train_id"] + ".pth")    # pylint: disable=no-member
        torch.save(learner.model.state_dict(), name)
    
try:
    run_one()
finally:
    if args.shutdown:
        # do whatever the right technique is for the platform(s) you run on.
        #os.system("sudo shutdown now")
        myname = platform.node()
        os.system(f"paperspace machine {myname} stop")
        
    

