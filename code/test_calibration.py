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
                        help="AGN for light curve calibration",
                        choices=AGN_NAMES)
    parser.add_argument("fltr", type=str,
                        help="Filter band for calibration",
                        choices=["u","B","g","V","r","i","z","all"])
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

    if args.fltr == 'all':
        args.fltr = None
    noprint=np.logical_not(args.verbose)
    # Create lightcurve model for this AGN
    model = AGNLCLib.AGNLCModel(PROJECTDIR,CONFIGDIR,args.agn)
    periods = [kk for kk in model.config().observation_params()['periods'].keys()]
    
    if args.show_outliers:
        for pp in periods:
            AGNLCLib.CalibrationOutlierPlot(model,pp,fltr=args.fltr,add_model=True,
                                     overwrite=True,noprint=noprint,delta=args.delta)

    if args.no_snr is False:
        for pp in periods:
            AGNLCLib.CalibrationSNR(model,pp,fltr=args.fltr,noprint=noprint)
