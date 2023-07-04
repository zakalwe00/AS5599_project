import os,re,argparse
import numpy as np
import pandas as pd
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

output_file = '{}/response.pdf'.format(output_dir)

# script to display the A/B priors for calibration
plt.rcParams.update({
    "font.family": "Sans", 
    "font.serif": ["DejaVu"],
    "figure.figsize":[12,9],
    "font.size": 14})

for i,fltr in enumerate(bessell_fltr_names):
    data_file = '{}/{}.csv'.format(output_dir,bessell_fltr_names[fltr])
    df = pd.read_csv(data_file)
    df_numpy = df.to_numpy()
    plt.plot(df_numpy[:,0],df_numpy[:,1],label='Bessell {}'.format(fltr))
    
for i,fltr in enumerate(sdss_fltr_names):
    data_file = '{}/{}.csv'.format(output_dir,sdss_fltr_names[fltr])
    df = pd.read_csv(data_file)
    df_numpy = df.to_numpy()
    plt.plot(df_numpy[:,0],df_numpy[:,1],label='SDSS {}'.format(fltr))
    
for i,fltr in enumerate(panstarrs_fltr_names):
    data_file = '{}/{}.csv'.format(output_dir,panstarrs_fltr_names[fltr])
    df = pd.read_csv(data_file)
    df_numpy = df.to_numpy()
    plt.plot(df_numpy[:,0],df_numpy[:,1],label='Pan-STARRS {}'.format(fltr))

plt.xlabel('$\lambda$ [nm]')
plt.ylabel('Transmission')
plt.yscale('log')
plt.ylim(1e-2,1.5)
plt.xlim(230,1200)
plt.legend(loc='center right')
plt.title("Filter band transmission response",fontsize=16)
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

    


                
