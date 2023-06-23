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

AGN = 'NGC_6814'

model = AGNLCLib.AGNLCModel(PROJECTDIR,CONFIGDIR,AGN)

#Remove the longer period, test only on the shorter
model.config().observation_params()['periods'].pop('Aug2022-Dec2022')

model.config().set_output_dir('{}/{}/output.test'.format(PROJECTDIR,AGN))

AGNLCLib.PyCCF(model,'g','z',overwrite=True)

        



                