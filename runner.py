# Sample code to set up and do a run by script
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

corine.set_corine_directory(os.environ['CORINE_DIR'])
windows_list = os.environ['LANDSAT_DIR'] + '/all_windows.csv'
neptune.init('denised/landcover')

wl = list(windows.from_file(windows_list))
(tr_list, val_list) = windows.randomized_split(wl,512)
#(tr_list, val_list) = windows.randomer_split(wl,512)
#tr_list = tr_list[:10000]

monitor = zoo.standard_monitor(n=100)
infra.set_defaults(bs=4, loss_func=zoo.SumQuadLoss())

learner = zoo.MultiUResNet.create(tr_list, val_list, title="what are we doing today?")
learner.fit(100, callbacks=[monitor])