import os,re,argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
#matplotlib.use('Agg')

def lognormal_pdf(x,sigma,mu):
    if x <= 0.0:
        return 0.0
    return ((x*sigma*np.sqrt(2.0*np.pi))**(-1))*np.exp(-0.5*(np.log(x-mu)/(sigma))**2)

def normal_pdf(x,sigma,mu):
    return ((sigma*np.sqrt(2.0*np.pi))**(-1))*np.exp(-0.5*((x-mu)/(sigma))**2)


output_dir = '{}/output'.format(str(Path(os.path.dirname(os.path.realpath(__file__))).parent))
if os.path.exists(output_dir) == False:
    print('Creating output dir {}'.format(output_dir))
    os.makedirs(output_dir)

output_file = '{}/calibration_priors.pdf'.format(output_dir)

if os.path.exists(output_file):
    print('Output file exists: {}, exiting.'.format(output_file))

# script to display the A/B priors for calibration

xx = np.linspace(-1.501,1.50,num=3000)
yy = [lognormal_pdf(x,0.02,0.0) for x in xx]
kk = [normal_pdf(x,0.5,0.0) for x in xx]


# create figure and axis objects with subplots()
fig,ax = plt.subplots()
plt.title("Lightcurve calibration priors",fontsize=16)
# make a plot
ax.plot(xx,yy,color="red")
# set y-axis label
ax.set_ylabel("$A_{s}$ log-normal prior PDF",
              color="red",
              fontsize=14)
# twin object second y-axis label
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(xx, kk,color="blue")
ax2.set_ylabel("$B_{s}$ normal prior PDF",color="blue",fontsize=14)
plt.savefig(output_file)
plt.show()

    


                
