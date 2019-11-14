# Sample code to set up and do a run by script
import argparse
import datetime
import platform
import pickle
from fastai import *
from fastai.basics import *
from multispectral import windows
from multispectral import tools
#import neptune
import infra
import zoo

parser = argparse.ArgumentParser()
parser.add_argument("--shutdown", help="Shutdown computer when complete", action='store_true')
parser.add_argument("--description", help="Description of run", default="runner.py")
parser.add_argument("--epochs", type=int, help="How many full epochs to run (default 1)", default=1)
parser.add_argument("--save", help="Save resulting model(s)", action='store_true' )
parser.add_argument("--cpu", help="Run on cpu", action='store_true')
parser.add_argument("--test_run", help="Run with a minimal dataset size to verify code works", action='store_true')
args = parser.parse_args()

# identity of directory to store things in.
gdrive_params = "-p 1J-laJXulHFxVQT3L3ZwRRdlyyGbDOhSo"

windows_list = os.environ['LANDSAT_DIR'] + '/all_windows.csv'
#neptune_project = neptune.init('denised/landcover')
infra.set_defaults(
    corine_directory=os.environ['CORINE_DIR'],
    traintracker_store=infra.TrainTrackerWebHook(os.environ['TRAINTRACKER_URI'])
    )

wl = list(windows.from_file(windows_list))
(tr_list, val_list) = windows.chunked_split(wl,512)
if args.test_run:
    tr_list = tr_list[:200]
    val_list = val_list[:20]

#ender = infra.TrainEnder()
#infra.set_defaults(train_end=4)
if args.cpu:
    infra.set_defaults(device=torch.device('cpu'))

logfilename = "runnerlog_{:%y%m%d_%H%M%S}.csv".format(datetime.datetime.now())

model_dir = Path(infra.defaults.model_directory)
def idname(learner):
    """Get the tracker id of the learner in a form suitable for use in a file name (i.e. remove the trailing space(s))"""
    return learner.parameters['train_id'].rstrip()

def save_sample(learner,data):
    """Save a little bit of data with predictions and targets, so we can explore without reloading"""
    predset = tools.get_prediction_set(learner,data).to_numpy()
    filename = model_dir / (idname(learner) + "_sample.pkl")
    with open(filename, 'wb') as fp:
        pickle.dump( predset, fp )
    return filename

def run_one(description=None, epochs=None, starting_from=None):
    description = description if description is not None else args.description
    epochs = epochs or args.epochs

    monitor_frequency = 80
    if args.test_run:
        monitor_frequency = 20
    monitor=infra.CycleHandler.create(n=monitor_frequency, callbacks=[
        infra.DiceMetric, 
        infra.GradientMetrics, 
        infra.CSVLogger(logfilename,'a')])
         
    learner = zoo.MultiUResNet.create(tr_list, val_list, callbacks=[], callback_fns=[monitor])
    if starting_from: # begin with existing weights
        learner.model.load_state_dict(torch.load(model_dir/starting_from))
    
    learner.fit(epochs, description=description)   # pylint: disable=unexpected-keyword-arg

    # save the model
    if args.save:
        name = model_dir / (idname(learner) + ".pth")
        torch.save(learner.model.state_dict(), name)

    # save a sample to look at.
    samplename = save_sample(learner, tr_list[:100])

    # upload log file and sample file
    # This requires gdrive to be installed (and of course you need your own account/folder to put things in)
    try:
        os.system(f"gdrive upload {gdrive_params} {logfilename}")
        os.system(f"gdrive upload {gdrive_params} {samplename}")
        os.system(f"mv {logfilename} logs")
    except Exception as e:
        # just print and continue
        print(e)
    
try:
    run_one()
finally:
    if args.shutdown:
        # do whatever the right technique is for the platform(s) you run on.
        #os.system("sudo shutdown now")
        myname = platform.node()
        os.system(f"paperspace machines stop --machineId {myname}")
        
    

