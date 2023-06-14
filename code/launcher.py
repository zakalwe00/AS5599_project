#!/minthome/hcornfield/.local/bin/python

import os,re,argparse
import pandas as pd
import numpy as np
import AGNLCLib

# setup global variables for use in the data pipeline (these can be overridden in environment)
PROJECTDIR = os.environ.get('PROJECTDIR','/minthome/hcornfield/git/AS5599_project')
#json files for project configuration
CONFIGDIR = os.environ.get('CONFIGDIR','/minthome/hcornfield/git/AS5599_project/config')


# objects we have data for are all subdirs of the project dir, removing the code dir
AGN_NAMES = [ agn.name for agn in os.scandir(PROJECTDIR) if agn.is_dir()
              and agn.name not in ['code','config'] and agn.name[0] != '.']

FUNCTION_MAPPING = {
    # PyROA functions adapted from https://github.com/FergusDonnan/PyROA
    'calibrate': AGNLCLib.InterCalibrateFilt,
    'fit': AGNLCLib.Fit,
    # Uses PYCCF code adapted from https://bitbucket.org/cgrier/python_ccf_code/src/master/
    'ccf': AGNLCLib.PyCCF }

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
    

        



                
