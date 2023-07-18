import os,re,argparse
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
#matplotlib.use('Agg')
#Telluric continuum extinction per airmass
#from Palomar multichannel reduction program.
ext = np.array([[2999,0.001],
                [3100,0.1758733],
                [3110,0.1980615],
                [3120,0.2234601],
                [3130,0.2491151],
                [3140,0.2708944],
                [3150,0.2900014],
                [3160,0.3064784],
                [3180,0.3388442],
                [3200,0.3684682],
                [3220,0.3926449],
                [3240,0.4164856],
                [3260,0.4357125],
                [3280,0.4470952],
                [3300,0.4655861],
                [3320,0.4795126],
                [3340,0.4911340],
                [3360,0.5025740],
                [3380,0.5119175],
                [3400,0.5209547],
                [3450,0.5435005],
                [3500,0.5628595],
                [3550,0.5802298],
                [3600,0.5981361],
                [3700,0.6286371],
                [3800,0.6570524],
                [3900,0.6829674],
                [4000,0.7072940],
                [4100,0.7277798],
                [4200,0.7467926],
                [4300,0.7648918],
                [4400,0.7805487],
                [4500,0.7950602],
                [4600,0.8068634],
                [4700,0.8180879],
                [4800,0.8279421],
                [4900,0.8371437],
                [5000,0.8441119],
                [5200,0.8558546],
                [5400,0.8653663],
                [5600,0.8725694],
                [5800,0.8782133],
                [6000,0.8822669],
                [6100,0.8863393],
                [6200,0.8928942],
                [6500,0.9120108],
                [6820,0.9272568],
                [6980,0.9323952],
                [7140,0.9375620],
                [7340,0.9418896],
                [7570,0.9462371],
                [7680,0.9479818],
                [8100,0.9541134],
                [8350,0.9567534],
                [8900,0.9611695],
                [9860,0.9673864],
                [10800,0.9709570],
                [12500,0.9727472]])

output_dir = '{}/output'.format(str(Path(os.path.dirname(os.path.realpath(__file__))).parent))
if os.path.exists(output_dir) == False:
    print('Creating output dir {}'.format(output_dir))
    os.makedirs(output_dir)

output_file = '{}/atmos.pdf'.format(output_dir)

# script to display the A/B priors for calibration
plt.rcParams.update({
    "font.family": "Sans", 
    "font.serif": ["DejaVu"],
    "figure.figsize":[8,6.5],
    "font.size": 14})

CCD_file = '{}/CCD.csv'.format(output_dir)
plt.plot(extinct[:,0],extinct[:,1],label='CCD',ls='dashed',color='black')

interp = interpolate.interp1d(df_ccd_numpy[:,0], df_ccd_numpy[:,1], kind="linear", fill_value="extrapolate")

for i,fltr in enumerate(bessell_fltr_names):
    data_file = '{}/{}.csv'.format(output_dir,bessell_fltr_names[fltr])
    df_numpy[fltr] = pd.read_csv(data_file).to_numpy()
    
for i,fltr in enumerate(sdss_fltr_names):
    data_file = '{}/{}.csv'.format(output_dir,sdss_fltr_names[fltr])
    df_numpy[fltr] = pd.read_csv(data_file).to_numpy()
    
for i,fltr in enumerate(panstarrs_fltr_names):
    data_file = '{}/{}.csv'.format(output_dir,panstarrs_fltr_names[fltr])
    df_numpy[fltr] = pd.read_csv(data_file).to_numpy()

for fltr in fltr_order:
    data = df_numpy[fltr]
    interp_ccd_response = interp(data[:,0])
    plt.plot(data[:,0],data[:,1],ls='dashed',color='grey',alpha=0.5)
    data[:,1] = data[:,1]*interp_ccd_response
    plt.plot(data[:,0],data[:,1],label=fltr)
    
plt.xlabel('$\lambda$ [nm]')
plt.ylabel('Transmission')
#plt.yscale('log')
plt.ylim(ymin=0.0)
plt.xlim(230,1120)
plt.legend(loc='upper right')
plt.title("Filter efficiency net of CCD sensitivity",fontsize=16)
# make a plot
#ax.plot(xx,yy,color="red")
# set y-axis label
#ax.set_ylabel("$A_{s}$ log-normal prior PDF",
#              color="red",
#              fontsize=14)
# twin object second y-axis label
#ax2=ax.twinx()
# make a plot with different y-axis using second axis object
#ax2.plot(xx, kk,color="blue")
#ax2.set_ylabel("$B_{s}$ normal prior PDF",color="blue",fontsize=14)
print('Writing {}'.format(output_file))
plt.savefig(output_file)
plt.show()

    


                
