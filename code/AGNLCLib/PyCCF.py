import sys
import argparse
import numpy as np
import pandas as pd
import csv
import matplotlib
from matplotlib import pyplot as plt
import scipy
# get the local copy of Utils
from . import Utils
import PYCCF
from scipy import stats 

# Uses the output from InterCalibrateFilt for two filter bands
# to run CCF_interp from the PyCCF codebase and output
# CCF/CCCD/CCPD data files
def PyCCF(model,fltr1,fltr2,overwrite=False):
    print('Running PyCCF interpolated cross correlation for filter bands {}, {}'.format(fltr1, fltr2))

    # references for convenience
    config = model.config()
    params = config.ccf_params()
    params.update(config.observation_params())
    
    ########################################
    ###Read in light curvedata
    ########################################    

    calib_file1 = '{}/{}_{}.dat'.format(config.output_dir(),config.agn_name(),fltr1)
    calib_file2 = '{}/{}_{}.dat'.format(config.output_dir(),config.agn_name(),fltr2)

    fltr1_all = pd.read_csv(calib_file1,
                            header=None,index_col=None,
                            quoting=csv.QUOTE_NONE,delim_whitespace=True).sort_values(0)

    fltr2_all = pd.read_csv(calib_file2,header=None,index_col=None,
                            quoting=csv.QUOTE_NONE,delim_whitespace=True).sort_values(0)

    # Time lag range to consider in the CCF (days).
    # Must be small enough that there is some overlap between light curves at that shift
    # (i.e., if the light curves span 80 days, these values must be less than 80 days).
    lag_range = params['lag_range']
    
    # lag range of data considered by year
    for period in params["periods"]:        
        centroidfile = '{}/Centroid_{}_{}_{}.dat'.format(config.output_dir(),period,fltr1,fltr2)
            
        #########################################
        ##Set Interpolation settings, user-specified
        #########################################
        # date range in data for light curve, typically a one-year observation run
        # from May to the following Feb
        mjd_range = params["periods"][period]["mjd_range"]

        fltr1_period = fltr1_all[np.logical_and(fltr1_all[0] >= mjd_range[0],
                                                fltr1_all[0] <= mjd_range[1])]
        fltr2_period = fltr2_all[np.logical_and(fltr2_all[0] >= mjd_range[0],
                                                fltr2_all[0] <= mjd_range[1])]

        mjd1,flux1,err1 = [col for col in fltr1_period.loc[:,0:2].T.to_numpy()]
        mjd2,flux2,err2 = [col for col in fltr2_period.loc[:,0:2].T.to_numpy()]

        median_cad1 = Utils.median_cadence(mjd1)
        median_cad2 = Utils.median_cadence(mjd2)
        print('Obs {0}, {3}+{4}, {1}+{2} unfiltered datapts, med cadence {5}+{6} (used for report)'.format(period,len(mjd1),len(mjd2),
                                                                                                           fltr1,fltr2,
                                                                                                           '{:.3f}'.format(median_cad1),
                                                                                                           '{:.3f}'.format(median_cad2)))
        fltr1_period = fltr1_period[fltr1_period[7] == False].loc[:,0:2]
        fltr2_period = fltr2_period[fltr2_period[7] == False].loc[:,0:2]

        mjd1,flux1,err1 = [col for col in fltr1_period.T.to_numpy()]
        mjd2,flux2,err2 = [col for col in fltr2_period.T.to_numpy()]

        median_cad1 = Utils.median_cadence(mjd1)
        median_cad2 = Utils.median_cadence(mjd2)
        print('Obs {0}, {3}+{4}, {1}+{2} filtered datapts, med cadence {5}+{6} (filtered sig_level={7})'.format(period,len(mjd1),len(mjd2),
                                                                                                               fltr1,fltr2,
                                                                                                               '{:.3f}'.format(median_cad1),
                                                                                                               '{:.3f}'.format(median_cad2),params["sig_level"]))
        if (Utils.check_file(centroidfile) == True) and (overwrite == False):
            print('Not running period {} {} vs {} calibration, file exists: {}'.format(period,fltr1,fltr2,centroidfile))
            continue

        # Interpolation time step (days). Must be less than the average cadence of the observations, but too small will introduce noise.
        # Consider the lowest median cadence from both curves and round down to nearest 1/20 days,
        # then take 1/5 of this.
        interp = params["periods"][period].get("Interp_Period",
                                               np.minimum(np.floor(median_cad1*4.0)*0.05,
                                                          np.floor(median_cad2*4.0)*0.05))
            
        
        nsim = params["MC_Iterations"]  #Number of Monte Carlo iterations for calculation of uncertainties

        thres = params["Centroid_Threshold"]
        
        print('Using lag_range={} days, interp={} days, nsim={}, thres={}'.format(lag_range,'{:.2f}'.format(interp),nsim,thres))

        # Do both FR/RSS sampling (1 = RSS only, 2 = FR only) 
        mcmode = 0

        # Choose the threshold for considering a measurement "significant".
        # sigmode = 0.2 will consider all CCFs with r_max <= 0.2 as "failed". See code for different sigmodes.
        sigmode = 0.2  

        ##########################################
        #Calculate lag with python CCF program
        ##########################################
        tlag_peak, status_peak, tlag_centroid, status_centroid, ccf_pack, max_rval, status_rval, pval = PYCCF.peakcent(mjd1,flux1,mjd2,flux2,
                                                                                                                       lag_range[0],lag_range[1],interp,thres=thres)
        tlags_peak, tlags_centroid, nsuccess_peak, nfail_peak, nsuccess_centroid, nfail_centroid, max_rvals, nfail_rvals, pvals = PYCCF.xcor_mc(mjd1,flux1,abs(err1),mjd2,flux2,abs(err2),
                                                                                                                                                lag_range[0],lag_range[1],interp,thres=thres,
                                                                                                                                                nsim=nsim,mcmode=mcmode,sigmode=sigmode)


        lag = ccf_pack[1]
        r = ccf_pack[0]

        # Z-score of 1 (1 s.d. above mean in normal dist)
        perclim = 84.1344746    

        ###Calculate the best peak and centroid and their uncertainties using the median of the
        ##distributions. 
        centau = stats.scoreatpercentile(tlags_centroid, 50)
        centau_uperr = (stats.scoreatpercentile(tlags_centroid, perclim))-centau
        centau_loerr = centau-(stats.scoreatpercentile(tlags_centroid, (100.-perclim)))
        print('Centroid, error: %10.3f  (+%10.3f -%10.3f)'%(centau, centau_loerr, centau_uperr))

        peaktau = stats.scoreatpercentile(tlags_peak, 50)
        peaktau_uperr = (stats.scoreatpercentile(tlags_peak, perclim))-centau
        peaktau_loerr = centau-(stats.scoreatpercentile(tlags_peak, (100.-perclim)))
        print('Peak, errors: %10.3f  (+%10.3f -%10.3f)'%(peaktau, peaktau_uperr, peaktau_loerr))


        ##########################################
        #Write results out to a file in case we want them later.
        ##########################################

        peakfile = '{}/Peak_{}_{}_{}.dat'.format(config.output_dir(),period,fltr1,fltr2)

        ccffile = '{}/CCF_{}_{}_{}.dat'.format(config.output_dir(),period,fltr1,fltr2)
        
        df = pd.DataFrame({'centroid':tlags_centroid,
                           'peak':tlags_peak})
        print('Writing {}'.format(centroidfile))
        df['centroid'].to_csv(centroidfile,
                              header=False,sep=' ',float_format='%25.15e',index=False,
                              quoting=csv.QUOTE_NONE,escapechar=' ')
        print('Writing {}'.format(peakfile))
        df['peak'].to_csv(peakfile,
                              header=False,sep=' ',float_format='%25.15e',index=False,
                              quoting=csv.QUOTE_NONE,escapechar=' ')
        df = pd.DataFrame({'lag':lag,
                           'r':r})
        print('Writing {}'.format(ccffile))
        df.to_csv(ccffile,
                  header=False,sep=' ',float_format='%25.15e',index=False,
                  quoting=csv.QUOTE_NONE,escapechar=' ')
        
        ##########################################
        #Plot the Light curves, CCF, CCCD, and CCPD
        ##########################################

        fig = plt.figure()
        fig.subplots_adjust(hspace=0.5, wspace = 0.1)
        fig.suptitle('Filter band {} centroid lag: {:5.2f} (+{:5.2f} -{:5.2f}) days'.format(fltr2, centau, centau_uperr, centau_loerr), fontsize = 15) 
        #Plot lightcurves
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.errorbar(mjd1,flux1,yerr = err1,marker ='.',ms=3.5,elinewidth=0.5,linestyle='dotted',color='k',label='Filter band {}'.format(fltr1))
        ax1_2 = fig.add_subplot(3, 1, 2, sharex = ax1)
        ax1_2.errorbar(mjd2,flux2,yerr=err2,marker ='.',ms=3.5,elinewidth=0.5,linestyle ='dotted',color ='k',label='Filter band {}'.format(fltr2))

        ax1.text(0.025, 0.825, 'ref: {}'.format(fltr1), fontsize = 10, transform = ax1.transAxes, color="red")
        ax1_2.text(0.025, 0.825, 'filter: {}'.format(fltr2), fontsize = 10, transform = ax1_2.transAxes, color="red")
        ax1.set_ylabel('Flux')
        ax1_2.set_ylabel('Flux')
        ax1_2.set_xlabel('MJD')

        #Plot CCF Information
        ax2 = fig.add_subplot(3, 3, 7)
        ax2.set_ylabel('CCF r')
        ax2.text(0.3, 0.75, 'CCF ', horizontalalignment = 'center', verticalalignment = 'center', transform = ax2.transAxes, fontsize = 10)
        ax2.set_ylim(-0.2, 1.2)
        # only one line may be specified; full height
        ax2.axvline(x=0,color='gray')
        ax2.axhline(y=0,color='gray')
        ax2.axhline(y=thres*np.max(r),color = 'blue',linestyle='dashed')
        ax2.plot(lag,r,color = 'k')
        
        ax3 = fig.add_subplot(3, 3, 8)
        ax3.axes.get_yaxis().set_ticks([])
        ax3.set_xlabel('MJD lag')
        ax3.text(0.3, 0.75, 'CCCD ', horizontalalignment = 'center', verticalalignment = 'center', transform = ax3.transAxes, fontsize = 10)
        n, bins, etc = ax3.hist(tlags_centroid, bins = 50, color = 'b')
        ax3.axvline(x=0,color = 'gray')

        ax4 = fig.add_subplot(3, 3, 9, sharex = ax3)
        ax4.set_ylabel('freq')
        ax4.yaxis.tick_right()
        ax4.yaxis.set_label_position('right') 
        ax4.text(0.3, 0.75, 'CCPD ', horizontalalignment = 'center', verticalalignment = 'center', transform = ax4.transAxes, fontsize = 10)
        ax4.hist(tlags_peak, bins = bins, color = 'b')
        ax4.axvline(x = 0,color = 'gray')

        plotccf = '{}/CCFResultsPlot_{}_{}_{}.pdf'.format(config.output_dir(),period,fltr1,fltr2)
        
        plt.savefig(plotccf)#, format = 'png', orientation = 'landscape', bbox_inches = 'tight')
        if matplotlib.get_backend() == 'TkAgg':
            plt.show()
        else:
            plt.close()

    return
