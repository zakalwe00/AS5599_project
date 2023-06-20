#!/minthome/hcornfield/.local/bin/python
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
model.config().set_output_dir('{}/{}/output.test'.format(PROJECTDIR,AGN))

# Artificially restrict the datapoints to consider
model.config().roa_params()['select_period'] = 'Mar2023-Current'
#model.config().roa_params()['plot_corner'] = False
#model.config().roa_params()['exclude_fltrs'] = ['z','u']
#model.config().roa_params()['Nsamples'] = 15000
#model.config().roa_params()['Nburnin'] = 10000

#model.config().set_output_dir('{}/{}/output.test'.format(PROJECTDIR,AGN))
#for fltr in model.config().fltrs():
#    if fltr != 'g':
#        AGNLCLib.PyCCF(model,'g',fltr,overwrite=True)
#AGNLCLib.Fit(model)
#AGNLCLib.FitPlot(model,overwrite=True)
AGNLCLib.ConvergencePlot(model,overwrite=True)
#AGNLCLib.ChainsPlot(model,overwrite=True)
#AGNLCLib.ChainsPlot(model,select='delta',overwrite=True)
#AGNLCLib.CornerPlot(model,overwrite=True)

        



                
