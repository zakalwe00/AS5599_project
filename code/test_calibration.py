import os,argparse,socket,sys
from itertools import count
import pandas as pd
import numpy as np
import AGNLCLib
import matplotlib
is_turgon = socket.gethostname() == 'turgon'
# Dont try to show graphs unless a terminal is attached and
# we're on a n appropriate server
if (sys.stdout.isatty() is False) or (is_turgon is False):
    matplotlib.use('Agg')


# setup global variables for use in the data pipeline (these can be overridden in environment)
HOMEDIR = os.environ['HOME']
TESTEXT = os.environ.get('TESTEXT','')
#json files for project configuration
PROJECTDIR = os.environ.get('PROJECTDIR','{}/git/AS5599_project'.format(HOMEDIR))
CONFIGDIR = os.environ.get('CONFIGDIR','{}/git/AS5599_project/config'.format(HOMEDIR))

# objects we have data for are all subdirs of the project dir, removing the code dir
AGN_NAMES = [ agn.name for agn in os.scandir(PROJECTDIR) if agn.is_dir()
              and agn.name not in ['code','config'] and agn.name[0] != '.']

                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("agn", type=str,
                        help="AGN for analysis",
                        choices=AGN_NAMES)
    parser.add_argument('-r', '--raw-plot',
                        action='store_true',
                        help='Plot raw uncalibrated telescope data by \
observing year')
    parser.add_argument('-s', '--show-outliers',
                        action='store_true',
                        help='Create plot of calibration model showing \
unclipped outliers which have been included')
    parser.add_argument('-d', '--delta',type=np.float64,
                        help='Delta value used for calibration, here used in the point\
density function to detect meaningful outliers. Otherwise median cadence used')
    parser.add_argument('-x', '--no-snr',
                        action='store_true',
                        help='Do not print out SNR for filter bands (\
included  by default)')
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Extra output logging for calculations \
such as number of values removed by clipping')
    args=parser.parse_args()
    
    noprint=np.logical_not(args.verbose)
    # Create lightcurve model for this AGN
    model = AGNLCLib.AGNLCModel(PROJECTDIR,CONFIGDIR,args.agn)
    periods = [kk for kk in model.config().observation_params()['periods'].keys()]
    
    if args.raw_plot:
        # run the raw filter plot for short sets of periods if necessary
        period_chunks = []
        for pp in range(0,len(periods),2):
            if pp == len(periods) - 1:
                if len(period_chunks) == 0:
                    period_chunks.append([periods[pp]])
                else:
                    period_chunks[-1].append(periods[pp])
            else:
                period_chunks.append([periods[pp],periods[pp+1]])
        old_period_map = model.config().observation_params()['periods']
        for pc in period_chunks:
            new_period_map = {}
            for ppc in pc:
                new_period_map[ppc] = old_period_map[ppc]
            model.config().observation_params()['periods'] = new_period_map
            AGNLCLib.ScopeRawPlot(model,noprint=noprint)
        model.config().observation_params()['periods'] = old_period_map

    if args.show_outliers:
        for pp in periods:
            AGNLCLib.CalibrationPlot(model,pp,add_model=True,overwrite=True,noprint=noprint)

    if args.no_snr is False:
        for pp in periods:
            AGNLCLib.CalibrationSNR(model,pp,noprint=noprint)
