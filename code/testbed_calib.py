import os,re,argparse
import pandas as pd
import numpy as np
import AGNLCLib
import matplotlib
#matplotlib.use('Agg')
HOMEDIR = os.environ['HOME']
# setup global variables for use in the data pipeline (these can be overridden in environment)
PROJECTDIR = os.environ.get('PROJECTDIR','{}/git/AS5599_project'.format(HOMEDIR))
#json files for project configuration
CONFIGDIR = os.environ.get('CONFIGDIR','{}/git/AS5599_project/config'.format(HOMEDIR))

#AGN = 'Fairall_9'
AGN = 'Fairall_9'

model = AGNLCLib.AGNLCModel(PROJECTDIR,CONFIGDIR,AGN,noprint=False)

#Remove the longer period, test only on the shorter
#model.config().observation_params()['periods'].pop('Aug2022-Dec2022')

#model.config().set_output_dir('{}/{}/output.largesigma_longmcmc_calib'.format(PROJECTDIR,AGN))
#model.config().set_output_dir('{}/{}/output.largesigma_calib'.format(PROJECTDIR,AGN))

#for fltr in model.config().calib_fltrs():
#for fltr in ["g","i"]:
for fltr in ["g"]:
#    AGNLCLib.InterCalibratePlot(model,fltr,'sig',overwrite=True)
    #for period in model.config().observation_params()['periods']:
    #    AGNLCLib.ScopeRawPlot(model,fltr,period,overwrite=True)
        

    periods = [kk for kk in model.config().observation_params()['periods'].keys()]
    period_chunks = []
    for pp in range(0,len(periods),2):
        if pp == len(periods) - 1:
            period_chunks[-1].append(periods[pp])
        else:
            period_chunks.append([periods[pp],periods[pp+1]])
    old_period_map = model.config().observation_params()['periods']
    for pc in period_chunks:
        new_period_map = {}
        for ppc in pc:
            new_period_map[ppc] = old_period_map[ppc]
        model.config().observation_params()['periods'] = new_period_map
        AGNLCLib.InterCalibratePlot(model,fltr,corner_plot=True,overwrite=False,mask_clipped=False)
                
