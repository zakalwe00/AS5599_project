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

model = AGNLCLib.AGNLCModel(PROJECTDIR,CONFIGDIR,AGN)

# We are running a test, divert output to the testbed
model.config().set_output_dir('{}/{}/output.old'.format(PROJECTDIR,AGN))

# Artificially restrict the datapoints to consider
#model.config().roa_params()['select_period']
#model.config().roa_params()['plot_corner'] = False
#model.config().roa_params()['exclude_fltrs'] = ['z','u']
#model.config().roa_params()['Nsamples'] = 25000
#model.config().roa_params()['Nburnin'] = 20000

#model.config().set_output_dir('{}/{}/output.test'.format(PROJECTDIR,AGN))
#for fltr in model.config().fltrs():
#    if fltr != 'g':
#        AGNLCLib.PyCCF(model,'g',fltr,overwrite=True)
#AGNLCLib.Fit(model)
for select_period in model.config().observation_params()['periods']:
    AGNLCLib.CalibrationSNR(model,select_period)
AGNLCLib.CalibrationSNR(model)
#AGNLCLib.ConvergencePlot(model,overwrite=True)
#AGNLCLib.ChainsPlot(model,overwrite=True)
#AGNLCLib.ChainsPlot(model,select='delta',overwrite=True)
#AGNLCLib.CornerPlot(model,overwrite=True)


                
