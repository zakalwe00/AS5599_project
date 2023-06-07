#!/minthome/hcornfield/.local/bin/python

import os, re
import argparse, itertools, sys, datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import PyROA

# setup global variables for use in the data pipeline

PROJECTDIR = os.environ.get('PROJECTDIR','/minthome/hcornfield/git/AS5599_project')
currentdatetime = datetime.datetime.now()
yyyymmdd = currentdatetime.strftime('%Y%m%d_%H%M%S')

# objects we have data for are all subdirs of the project dir, removing the code dir
AGN_NAMES = [ agn.name for agn in os.scandir(PROJECTDIR) if agn.is_dir()
              and agn.name != 'code' and agn.name[0] != '.']

def check_and_create_dir(check_dir):
    if os.path.exists(check_dir) and os.path.isfile(check_dir):
        # directory exists but its a file
        os.remove(check_dir)
    if os.path.exists(check_dir) == False:
        print('Creating directory {}'.format(check_dir))
        os.mkdir(check_dir)
    return
        
def load_lco_lightcurves(agn):
    # Load Las Cumbres Observatory data from AVA https://www.alymantara.com/ava/
    # if available from the project directory for the AGN
    lco_lc_file = '{}/{}/LCO/AVA_{}_lco.csv'.format(PROJECTDIR,agn,agn)
    if os.path.isfile(lco_lc_file) == False:
        raise Exception('LCO lightcurve does not exist in {}'.format(lco_lc_file))
    print('Found LCO lightcurve file {}'.format(lco_lc_file))
    return lco_lc_file

def write_scope_filter_data(agn,obs_file):
    # split LCO file data for this AGN into records by telescope/filter (spectral band)
    obs = pd.read_csv(obs_file)
    scopes = np.unique(obs.Tel)
    print('Found telescope list {}'.format(','.join(scopes)))
    fltrs = np.unique(obs.Filter)
    print('Found filter list {}'.format(','.join(fltrs)))

    # (possibly) first time output directory is touched
    output_dir = '{}/{}/output'.format(PROJECTDIR, agn)
    check_and_create_dir(output_dir)
    
    # prepare data file per telescope/filter if not already done
    for scope in scopes:
        for fltr in fltrs:
            output_fn = '{}/{}_{}_{}.dat'.format(output_dir, fltr, agn, scope)
            if os.path.exists(output_fn) == False:
                # Select time/flux/error per scope and filter
                obs_scope = obs[obs['Tel'] == scope]
                try:
                    obs_scope_fltr  = obs_scope[obs_scope['Filter'] == fltr].loc[:,['MJD','Flux','Error']]
                except KeyError:
                    print('Filter {} not found for telescope {}'.format(fltr, scope))
                    continue
                print('Writing file {}'.format(output_fn))
                obs_scope_fltr.to_csv(output_fn, sep=' ', index=False, header=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agn", type=str,
                        default=None, help="AGN to run analysis pipeline \
                        for in order to determine metrics",
                        choices=AGN_NAMES)
    args=parser.parse_args()

    # Load and organise lightcurve data
    lco_lc_file = load_lco_lightcurves(args.agn)
    write_scope_filter_data(args.agn,lco_lc_file)
    
            
        
# From PyROA project Inter-calibration example, now
# each lightcurve for each telescope is saved as a .dat
# file where columns are time, flux, flux_err

#for fltr in fltrs:
#    print('Running PyROA InterCalibrate for {} filter {}'.format(qname, fltr))
#    fit = PyROA.InterCalibrate(output_dir, qname, fltr, scopes)


                
