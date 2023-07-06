import os,re,argparse
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
#matplotlib.use('Agg')


output_dir = '{}/output'.format(str(Path(os.path.dirname(os.path.realpath(__file__))).parent))
if os.path.exists(output_dir) == False:
    print('Creating output dir {}'.format(output_dir))
    os.makedirs(output_dir)

bessell_fltr_names = { 'V': 'BSSL-VX-022',
                       'B': 'BSSL-BX-004' }
sdss_fltr_names = { 'g\'': 'SDSS-g',
                    'i\'': 'SDSS-i',
                    'r\'': 'SDSS-r',
                    'u\'': 'SDSS-u' }
panstarrs_fltr_names = { '$z_{s}$': 'SDSS-z' }

fltr_order = ['u\'','B','g\'','V','r\'','i\'','$z_{s}$']

output_file = '{}/response.pdf'.format(output_dir)

df_numpy = {}

# script to display the A/B priors for calibration
plt.rcParams.update({
    "font.family": "Sans", 
    "font.serif": ["DejaVu"],
    "figure.figsize":[8,6.5],
    "font.size": 14})

CCD_file = '{}/CCD.csv'.format(output_dir)
df_ccd_numpy = pd.read_csv(CCD_file).to_numpy()
#convert from percentage
df_ccd_numpy[:,1] = df_ccd_numpy[:,1]/100.0
plt.plot(df_ccd_numpy[:,0],df_ccd_numpy[:,1],label='CCD',ls='dashed',color='black')

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

    


                
