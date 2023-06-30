import os,re,argparse
import pandas as pd
import numpy as np
import AGNLCLib
import matplotlib
#matplotlib.use('Agg')

# setup global variables for use in the data pipeline (these can be overridden in environment)
PROJECTDIR = os.environ.get('PROJECTDIR','/minthome/hcornfield/git/AS5599_project')
#json files for project configuration
CONFIGDIR = os.environ.get('CONFIGDIR','/minthome/hcornfield/git/AS5599_project/config')

AGN = 'NGC_6814'

model = AGNLCLib.AGNLCModel(PROJECTDIR,CONFIGDIR,AGN,noprint=False)

#Remove the longer period, test only on the shorter
#model.config().observation_params()['periods'].pop('Aug2022-Dec2022')

#model.config().set_output_dir('{}/{}/output.largesigma_longmcmc_calib'.format(PROJECTDIR,AGN))
#model.config().set_output_dir('{}/{}/output.largesigma_calib'.format(PROJECTDIR,AGN))

for fltr in model.config().calib_fltrs():
#for fltr in ["g","i"]:
#for fltr in ["i"]:
#    AGNLCLib.InterCalibratePlot(model,fltr,'sig',overwrite=True)
    for period in model.config().observation_params()['periods']:
        AGNLCLib.ScopeRawPlot(model,fltr,period,overwrite=True)
        



                
