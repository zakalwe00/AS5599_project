import os,argparse,socket,sys
import pandas as pd
import numpy as np
import AGNLCLib
import matplotlib
if sys.stdout.isatty() is False:
    matplotlib.use('Agg')


# setup global variables for use in the data pipeline (these can be overridden in environment)
HOMEDIR = os.environ['HOME']
TESTEXT = os.environ.get('TESTEXT','')
#json files for project configuration
PROJECTDIR = os.environ.get('PROJECTDIR','{}/git/AS5599_project'.format(HOMEDIR,TESTEXT))
CONFIGDIR = os.environ.get('CONFIGDIR','{}/git/AS5599_project{}/config'.format(HOMEDIR,TESTEXT))

# objects we have data for are all subdirs of the project dir, removing the code dir
AGN_NAMES = [ agn.name for agn in os.scandir(PROJECTDIR) if agn.is_dir()
              and agn.name not in ['code','config'] and agn.name[0] != '.']

FUNCTION_MAPPING = {
    # PyROA functions adapted from https://github.com/FergusDonnan/PyROA
    'calibrate': AGNLCLib.InterCalibrateFilt,
    'calibrate_filt_plot': AGNLCLib.InterCalibratePlot, # Corner plot, LC plot by filter
    'calibrate_snr': AGNLCLib.CalibrationSNR,
    'calibrate_plot': AGNLCLib.CalibrationPlot,
    'roa': AGNLCLib.Fit,
    'roa_plot': AGNLCLib.FitPlot,
    'roa_conv_plot': AGNLCLib.ConvergencePlot,
    'roa_chains_plot': AGNLCLib.ChainsPlot,
    'roa_corner_plot': AGNLCLib.CornerPlot }

is_turgon = socket.gethostname() == 'turgon'
# Uses PYCCF code adapted from https://bitbucket.org/cgrier/python_ccf_code/src/master/
# installed on turgon but not AWS
if is_turgon:
    FUNCTION_MAPPING['ccf'] = AGNLCLib.PyCCF

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("agn", type=str,
                        help="AGN for analysis",
                        choices=AGN_NAMES)
    parser.add_argument("function", type=str,
                        help="Analysis function to run with comma deliminated args, format example <func_name>:arg1,arg2.\
Available functions are {}".format(','.join([xx for xx in FUNCTION_MAPPING.keys()])))

    args=parser.parse_args()

    fargs = args.function.split(':')
    function = FUNCTION_MAPPING[fargs[0]]
    function_args = []
    if len(fargs) == 2:
        function_args = fargs[1].split(',')
    elif len(fargs) > 2:
        raise Exception('Unexpected format for function argument {}'.format(args.function))

    # Create lightcurve model for this AGN
    model = AGNLCLib.AGNLCModel(PROJECTDIR,CONFIGDIR,args.agn)
    # run function
    function(model,*function_args)
    

        



                
