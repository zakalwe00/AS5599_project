import sys
import argparse
import numpy as np
import pandas as pd
import csv
from matplotlib import pyplot as plt
import scipy
from PyROA import Utils
import PYCCF
from scipy import stats 

# Uses the output from InterCalibrateFilt for two filter bands
# to run CCF_interp from the PyCCF codebase and output
# CCF/CCCD/CCPD data files
def PyCCF(model,fltr1,fltr2):
    print('Running PyCCF intercorrelation for filter bands {}, {}'.format(fltr1, fltr2))

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

    sigma_limit = params['sigma_limit']
    # Time lag range to consider in the CCF (days).
    # Must be small enough that there is some overlap between light curves at that shift
    # (i.e., if the light curves span 80 days, these values must be less than 80 days).
    lag_range = params['lag_range']
    
    # lag range of data considered by year
    for period in params["periods"]:        
        centroidfile = '{}/Centroid_{}_{}_{}.dat'.format(config.output_dir(),period,fltr1,fltr2)
        if Utils.check_file(centroidfile) == True:
            print('Not running period {} {} vs {} calibration, file exists: {}'.format(period,fltr1,fltr2,centroidfile))
            break
            
        #########################################
        ##Set Interpolation settings, user-specified
        #########################################
        # date range in data for light curve, typically a one-year observation run
        # from May to the following Feb
        mjd_range = params["periods"][period]["mjd_range"]

        # filter data with sigma > sigma_limit * mean_err for this date range
        fltr1_period = fltr1_all[np.logical_and(fltr1_all[0] > mjd_range[0],
                                                fltr1_all[0] < mjd_range[1])].loc[:,0:2]
        mean_err1 = np.mean(fltr1_period.loc[:,2])
        mjd1,flux1,err1 = [col for col in fltr1_period[fltr1_period.loc[:,2] < mean_err1 * sigma_limit].T.to_numpy()]
        fltr2_period = fltr2_all[np.logical_and(fltr2_all[0] > mjd_range[0],
                                                fltr2_all[0] < mjd_range[1])].loc[:,0:2]
        mean_err2 = np.mean(fltr2_period.loc[:,2])
        mjd2,flux2,err2 = [col for col in fltr2_period[fltr2_period.loc[:,2] < mean_err2 * sigma_limit].T.to_numpy()]

        median_cad1 = Utils.median_cadence(mjd1)
        median_cad2 = Utils.median_cadence(mjd2)
        print('Obs {0}, {6}+{7}, {1}+{2} datapts after filter (err < {3}*{5} + {4}*{5}) med cadence {8}+{9}'.format(period,len(mjd1),len(mjd2),
                                                                                                                    '{:.5f}'.format(mean_err1),
                                                                                                                    '{:.5f}'.format(mean_err2),
                                                                                                                    sigma_limit,fltr1,fltr2,
                                                                                                                    '{:.3f}'.format(median_cad1),
                                                                                                                    '{:.3f}'.format(median_cad2)))
        
        # Interpolation time step (days). Must be less than the average cadence of the observations, but too small will introduce noise.
        # Consider the lowest median from both curves and ound down to nearest 1/20 days.
        interp = params["periods"][period].get("med_cadence",
                                                   np.minimum(np.floor(median_cad1*20.0)*0.05,
                                                              np.floor(median_cad2*20.0)*0.05))
        print('Using lag_range={} days, interp={} days'.format(lag_range,'{:.2f}'.format(interp)))
        
        nsim = params["Niter"]  #Number of Monte Carlo iterations for calculation of uncertainties

        mcmode = 0                  #Do both FR/RSS sampling (1 = RSS only, 2 = FR only) 
        #Choose the threshold for considering a measurement "significant".
        # sigmode = 0.2 will consider all CCFs with r_max <= 0.2 as "failed". See code for different sigmodes.
        sigmode = 0.2  

        ##########################################
        #Calculate lag with python CCF program
        ##########################################
        tlag_peak, status_peak, tlag_centroid, status_centroid, ccf_pack, max_rval, status_rval, pval = PYCCF.peakcent(mjd1, flux1, mjd2, flux2,
                                                                                                                       lag_range[0], lag_range[1], interp)
        tlags_peak, tlags_centroid, nsuccess_peak, nfail_peak, nsuccess_centroid, nfail_centroid, max_rvals, nfail_rvals, pvals = PYCCF.xcor_mc(mjd1, flux1, abs(err1),
                                                                                                                                                mjd2, flux2, abs(err2),
                                                                                                                                                lag_range[0], lag_range[1], interp,
                                                                                                                                                nsim = nsim, mcmode=mcmode, sigmode =sigmode)

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
        fig.subplots_adjust(hspace=0.2, wspace = 0.1)

        #Plot lightcurves
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.errorbar(mjd1, flux1, yerr = err1, marker = '.', linestyle = 'dotted', color = 'k', label = 'Filter band {}'.format(fltr1))
        ax1_2 = fig.add_subplot(3, 1, 2, sharex = ax1)
        ax1_2.errorbar(mjd2, flux2, yerr = err2, marker = '.', linestyle = 'dotted', color = 'k', label = 'Filter band {}'.format(fltr2))

        ax1.text(0.025, 0.825, fltr1, fontsize = 15, transform = ax1.transAxes)
        ax1_2.text(0.025, 0.825, fltr2, fontsize = 15, transform = ax1_2.transAxes)
        ax1.set_ylabel('LC 1 Flux')
        ax1_2.set_ylabel('LC 2 Flux')
        ax1_2.set_xlabel('MJD')

        #Plot CCF Information
        xmin, xmax = -99, 99
        ax2 = fig.add_subplot(3, 3, 7)
        ax2.set_ylabel('CCF r')
        ax2.text(0.2, 0.85, 'CCF ', horizontalalignment = 'center', verticalalignment = 'center', transform = ax2.transAxes, fontsize = 16)
        ax2.set_xlim(xmin, xmax)
        ax2.set_ylim(-1.0, 1.0)
        ax2.plot(lag, r, color = 'k')
        
        ax3 = fig.add_subplot(3, 3, 8, sharex = ax2)
        ax3.set_xlim(xmin, xmax)
        ax3.axes.get_yaxis().set_ticks([])
        ax3.set_xlabel('Centroid Lag: %5.1f (+%5.1f -%5.1f) days'%(centau, centau_uperr, centau_loerr), fontsize = 15) 
        ax3.text(0.2, 0.85, 'CCCD ', horizontalalignment = 'center', verticalalignment = 'center', transform = ax3.transAxes, fontsize = 16)
        n, bins, etc = ax3.hist(tlags_centroid, bins = 50, color = 'b')

        ax4 = fig.add_subplot(3, 3, 9, sharex = ax2)
        ax4.set_ylabel('N')
        ax4.yaxis.tick_right()
        ax4.yaxis.set_label_position('right') 
        #ax4.set_xlabel('Lag (days)')
        ax4.set_xlim(xmin, xmax)
        ax4.text(0.2, 0.85, 'CCPD ', horizontalalignment = 'center', verticalalignment = 'center', transform = ax4.transAxes, fontsize = 16)
        ax4.hist(tlags_peak, bins = bins, color = 'b')
        
        plotccf = '{}/CCFResultsPlot_{}_{}_{}.png'.format(config.output_dir(),period,fltr1,fltr2)
        
        plt.savefig(plotccf, format = 'png', orientation = 'landscape', bbox_inches = 'tight') 
        plt.close(fig)

    return
