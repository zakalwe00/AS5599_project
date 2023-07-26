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
    "figure.figsize":[9,5],
    "font.size": 14})

CCD_file = '{}/CCD.csv'.format(output_dir)
df_ccd_numpy = pd.read_csv(CCD_file).to_numpy()
#convert from percentage
df_ccd_numpy[:,1] = df_ccd_numpy[:,1]/100.0
plt.plot(10*df_ccd_numpy[:,0],df_ccd_numpy[:,1],label='CCD',ls='dashed',color='black')

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
    plt.plot(10*data[:,0],data[:,1],ls='dashed',color='grey',alpha=0.5)
    data[:,1] = data[:,1]*interp_ccd_response
    plt.plot(10*data[:,0],data[:,1],label=fltr)
    
plt.xlabel('$\lambda$ ($\AA$)')
plt.ylabel('Transmittance')
plt.yscale('log')
plt.ylim(0.005,1.5)
plt.xlim(2300,11200)
plt.legend(loc='upper right')

print('Writing {}'.format(output_file))
plt.savefig(output_file)
plt.show()

    


                
