#!/minthome/hcornfield/.local/bin/python

import os, re
import pandas as pd
#import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import PyROA
import csv

# Set Seyfert 1 Galaxy name in this case (will be argument to this script?)
qname = 'NGC_6814'

output_dir = '{}/../output/'.format(os.getcwd())
            
#Only fit a subsection of lightcurves- those available from calibration
#'g' filter is suggested as having highest signal-to-noise (from paper) https://arxiv.org/pdf/2302.09370.pdf
#and first filter is taken as delay reference time
fltrs=['V','g','B']

#set this to a reasonable guess to improve 'burn-in' time
init_tau = [5.0, 10.0]

#From Example_Usage notebook
#Priors are uniform where the limits must be specified in the following way:
#priors = [[A_lower, A_upper], [B_lower, B_upper], [tau_lower, tau_upper],[delta_lower, delta_upper], [sig_lower, sig_upper]]
priors = [[0.5, 2.0],[0.5, 2.0], [0.0, 20.0], [0.05, 5.0], [0.0, 10.0]]

# All the calibrated lightcurves pre-fitting on the same plot
calib_curve_plot = '{}/Calibrated_LCs.pdf'.format(output_dir)
if os.path.exists(calib_curve_plot) == False:
    data=[]
    plt.style.use(['seaborn'])
    plt.rcParams.update({
        "font.family": "Sans",  
        "font.serif": ["DejaVu"],
        "figure.figsize":[40,15],
        "font.size": 40})
    fig, axs = plt.subplots(len(fltrs),sharex=True)
    fig.suptitle('{} Calibrated light curves'.format(qname))
    for i,fltr in enumerate(fltrs):
        calib_file = '{}/{}_{}.dat'.format(output_dir,qname,fltr)
        data.append(pd.read_csv(calib_file,
                                header=None,index_col=None,
                                quoting=csv.QUOTE_NONE,delim_whitespace=True))
        mjd = data[i][0]
        flux = data[i][1]
        err = data[i][2]
        axs[i].errorbar(mjd, flux , yerr=err, ls='none', marker=".", ms=3.5, elinewidth=0.5)
        axs[i].set_ylabel('{} filter flux'.format(fltr))

    axs[-1].set_xlabel('Time (days, MJD)')

    plt.savefig(calib_curve_plot)
    plot.show()
    
#Has delay_dist=False set
fit = PyROA.Fit(output_dir,qname,fltrs,priors,add_var=True,init_tau=init_tau,Nsamples=10000, Nburnin=5000)
