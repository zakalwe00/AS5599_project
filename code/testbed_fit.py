#!/minthome/hcornfield/.local/bin/python
import os,re,argparse
import pandas as pd
import numpy as np
import AGNLCLib
import matplotlib
matplotlib.use('Agg')

# setup global variables for use in the data pipeline (these can be overridden in environment)
PROJECTDIR = os.environ.get('PROJECTDIR','/minthome/hcornfield/git/AS5599_project')
#json files for project configuration
CONFIGDIR = os.environ.get('CONFIGDIR','/minthome/hcornfield/git/AS5599_project/config')

AGN = 'Fairall_9'

model = AGNLCLib.AGNLCModel(PROJECTDIR,CONFIGDIR,AGN)

#Remove the longer period, test only on the shorter
#model.config().observation_params()['periods'].pop('Aug2022-Dec2022')

# just run with 'g', 'i', 'r'
model.config().fltrs().pop('V')
model.config().fltrs().pop('B')
model.config().fltrs().pop('z')
model.config().fltrs().pop('u')

model.config().set_output_dir('{}/{}/output.test'.format(PROJECTDIR,AGN))

for fltr in 
    if fltr != 'g':
        AGNLCLib.Fit(model)

        



                
