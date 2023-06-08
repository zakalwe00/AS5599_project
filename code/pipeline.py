#!/minthome/hcornfield/.local/bin/python

import os,re,argparse,datetime
import pandas as pd
import numpy as np
import PyROA

# setup global variables for use in the data pipeline

PROJECTDIR = os.environ.get('PROJECTDIR','/minthome/hcornfield/git/AS5599_project')
currentdatetime = datetime.datetime.now()
yyyymmdd = currentdatetime.strftime('%Y%m%d_%H%M%S')

# objects we have data for are all subdirs of the project dir, removing the code dir
AGN_NAMES = [ agn.name for agn in os.scandir(PROJECTDIR) if agn.is_dir()
              and agn.name != 'code' and agn.name[0] != '.']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("agn", type=str,
                        help="AGN to run analysis pipeline for in order to determine metrics",
                        choices=AGN_NAMES)
    parser.add_argument("-c", "--calibrate", action='store_true',
                        default=None,
                        help="Calibrate LCO lightcurve data")
    parser.add_argument("-cp", "--calibration-plot", action='store_true',
                        default=False,
                        help="Include a calibration corner plot in the output")
    parser.add_argument("-ccp", "--calibration-corner-plot", action='store_true',
                        default=False,
                        help="Include a calibration corner plot in the output")

    args=parser.parse_args()

    # Create lightcurve model for this AGN
    model = PyROA.LCModel(PROJECTDIR,args.agn)

    # Calibrate LCO data if necessary
    if args.calibrate:

        for fltr in model.fltrs():
            print('Running PyROA InterCalibrate for {} filter {}'.format(args.agn, fltr))
            model.InterCalibrateFilt(fltr)
            if args.calibration_plot:
                model.InterCalibrateFiltPlot(fltr,args.calibration_corner_plot)
    
            
        



                
