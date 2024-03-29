import os,re,argparse
import pandas as pd
import numpy as np
import AGNLCLib
import matplotlib
#matplotlib.use('Agg')

HOME = os.environ['HOME']
# setup global variables for use in the data pipeline (these can be overridden in environment)
PROJECTDIR = os.environ.get('PROJECTDIR','{}/git/AS5599_project'.format(HOME))
#json files for project configuration
CONFIGDIR = os.environ.get('CONFIGDIR','{}/git/AS5599_project/config'.format(HOME))

#AGN = 'Fairall_9_Jun18-Feb19'
#select_period = 'Jun18-Feb19'
AGN = 'NGC_1365_May22-Mar23'
select_period = 'May22-Mar23'
#AGN = 'Fairall_9_May20-Feb21'
#select_period = 'May20-Feb21'

model = AGNLCLib.AGNLCModel(PROJECTDIR,CONFIGDIR,AGN)

#Remove the longer period, test only on the shorter
#model.config().observation_params()['periods'].pop('Aug2022-Dec2022')

#model.config().set_output_dir('{}/{}/output.test'.format(PROJECTDIR,AGN))

#for fltr in model.config().calib_fltrs():
#for fltr in ["i"]:
#    for period in model.config().observation_params()['periods']:
#        AGNLCLib.ScopeRawPlot(model,fltr,period,overwrite=False)
#    AGNLCLib.InterCalibratePlot(model,fltr,overwrite=False)
#

#AGNLCLib.ConvergencePlot(model,select_period,overwrite=True)
#AGNLCLib.FluxFlux(model,select_period,overwrite=True)
AGNLCLib.LagSpectrum(model,select_period,overwrite=False,add_ccf=True)



                
