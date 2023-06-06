#!/minthome/hcornfield/.local/bin/python

import os, re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import PyROA

# Set Seyfert 1 Galaxy name in this case (will be argument to this script?)
qname = 'Fairall_9'
obs_file = ''

# Load Las Cumbres Observatory data for this object from AVA https://www.alymantara.com/ava/

with os.scandir('{}/../LCO'.format(os.getcwd())) as lco_files:
    for entry in lco_files:
        if entry.is_file():
            if entry.name == 'AVA_{}_lco.csv'.format(qname):
                obs_file = entry.path
                lco_files.close()
                break

output_dir = '{}/../output/'.format(os.getcwd())
            
obs = pd.read_csv(obs_file)
scopes = np.unique(obs.Tel)
fltrs = np.unique(obs.Filter)

# prepare data file per scope / per filter
for scope in scopes:
    for fltr in fltrs:
        output_fn = '{}{}_{}_{}.dat'.format(output_dir, qname, fltr, scope)
        if os.path.exists(output_fn) == False:
            # Select time/flux/error per scope and filter
            obs_scope = obs[obs['Tel'] == scope]
            try:
                obs_scope_fltr  = obs_scope[obs_scope['Filter'] == fltr].loc[:,['MJD','Flux','Error']]
            except KeyError:
                print('Filter {} not found for telescope {}'.format(fltr, scope))
                continue
            
            obs_scope_fltr.to_csv(output_fn, sep=' ', index=False, header=False)
        
# From PyROA project Inter-calibration example, now
# each lightcurve for each telescope is saved as a .dat
# file where columns are time, flux, flux_err

for fltr in fltrs:
    print('Running PyROA InterCalibrate for {} filter {}'.format(qname, fltr))
    fit = PyROA.InterCalibrate(output_dir, qname, fltr, scopes)


                