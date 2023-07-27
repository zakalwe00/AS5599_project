import os,json
# get the local copy of Utils
from . import Utils
from multiprocessing import Pool
from itertools import chain
from tabulate import tabulate
import corner
import numpy as np
import pandas as pd
import csv
import pickle
import matplotlib.ticker as mtick
import matplotlib.ticker as ticker
import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
matplotlib.rc('xtick', labelsize=12) 
matplotlib.rc('ytick', labelsize=12) 
import scipy.interpolate as interpolate
import scipy.special as special
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import quad


########################################
# Diagnostic Graphs                    #
########################################
def ScopeRawPlot(model,fltr,overwrite=False,noprint=True):

    print('Running PyROA ScopeRawPlot for filter {}'.format(fltr))

    config = model.config()

    # set up scopes to be used for calibration
    scopes = config.scopes()
    exclude_scopes = config.calibration_params().get("exclude_scopes",[])

    print('{} filter band data available from {} with {} to be excluded'.format(fltr,scopes,exclude_scopes))

    # set up the local variables
    data = []
    scopes_array = []
    
    # read original LCs by scope
    for scope in scopes:
        scope_file = '{}/{}_{}_{}.dat'.format(config.output_dir(),config.agn_name(),fltr,scope)
        #Check if file is empty
        dd = np.loadtxt(scope_file)
        if dd.shape[0] != 0:
            data.append(dd)
    
    plt.rcParams.update({
        "font.family": "Sans", 
        "font.serif": ["DejaVu"],
        "figure.figsize":[20,10],
        "font.size": 14})

    period_to_mjd_range = config.observation_params()['periods']

    gs = gridspec.GridSpec(len(period_to_mjd_range.keys()), 1) 

    for i,period in enumerate(period_to_mjd_range):
        ax0 = plt.subplot(gs[i])
        mjd_range = period_to_mjd_range[period]['mjd_range']
        for j in range(len(data)):
            mask = np.logical_and(data[j][:,0] >= mjd_range[0],data[j][:,0] <= mjd_range[1])
            mjd = data[j][mask,0]
            flux = data[j][mask,1]
            err = data[j][mask,2]
            ax0.errorbar(mjd, flux, yerr=err, ls='none', marker=".", label=str(scopes[j]))
        ax0.set_ylabel('Flux (mJy) {}'.format(period))
        # Shrink x=axis by 10% in order to fit the legend on the right
        box = ax0.get_position()
        ax0.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        if i == 0:
            ax0.set_title('{} uncalibrated light curves for by telescope (filter band {})'.format(config.agn(),fltr))
            # Put a legend containing telescope names to the right in the cleared space
            ax0.legend(title="Telescope list",loc='center left', bbox_to_anchor=(1.0, 0.05))
    ax0.set_xlabel('Time (days, MJD)')
    periods = [kk for kk in model.config().observation_params()['periods'].keys()]
    if len(periods) > 1:
        output_file = '{}/{}_Raw_Plot_{}_{}.pdf'.format(config.output_dir(),fltr,periods[0],periods[-1])
    else:
        output_file = '{}/{}_Raw_Plot_{}.pdf'.format(config.output_dir(),fltr,periods[0])
    if (os.path.exists(output_file) == True) and (overwrite == False):
        print('Not writing PyROA ScopeRaw, file exists: {}'.format(output_file))
    else:
        print('Writing raw scope data plot {}'.format(output_file))
        plt.savefig(output_file)
    if matplotlib.get_backend() == 'TkAgg':
        plt.show()
    else:
        plt.close()
    


def InterCalibratePlot(model,fltr,select='A',corner_plot=True,overwrite=False,mask_clipped=False,noprint=True):

    print('Running PyROA InterCalibratePlot for filter {}'.format(fltr))

    # references for convenience
    config = model.config()
    calib_params = config.calibration_params()
    sig_level = calib_params['sig_level']
    constrain_dates = calib_params.get("constrain_dates",None)
    
    # set up scopes to be used for calibration
    scopes = config.scopes()
    exclude_scopes = calib_params.get("exclude_scopes",[])
    scopes = [scope for scope in scopes if scope not in exclude_scopes]
    print('Plotting calibrated lightcurve for {} with {} excluded'.format(scopes,exclude_scopes))

    # set up the local variables
    data = []
    scopes_array = []

    # read original LCs by scope
    for scope in scopes:
        scope_file = '{}/{}_{}_{}.dat'.format(config.output_dir(),config.agn_name(),fltr,scope)
        #Check if file is empty
        dd = np.loadtxt(scope_file)
        if dd.shape[0] != 0:
            if constrain_dates is not None:
                dd = dd[np.logical_and(dd[:,0] >= constrain_dates[0],dd[:,0] <= constrain_dates[1]),:]
                data.append(dd)
                scopes_array.append([scope]*np.loadtxt(scope_file).shape[0])
            
    scopes_array = [item for sublist in scopes_array for item in sublist]
    scopes = np.unique(scopes_array)
    
    # read calibration file which should now exist
    calib_file = '{}/{}_{}.dat'.format(config.output_dir(),config.agn_name(),fltr)
    Utils.check_file(calib_file,exit=True)    
    df = pd.read_csv(calib_file,
                     header=None,index_col=None,
                     quoting=csv.QUOTE_NONE,delim_whitespace=True).sort_values(0)
    
    filehandler = open('{}/{}_calib_samples_flat.obj'.format(config.output_dir(),fltr),"rb")
    samples_flat = pickle.load(filehandler)
        
    filehandler = open('{}/{}_calib_samples.obj'.format(config.output_dir(),fltr),"rb")
    samples = pickle.load(filehandler)

    filehandler = open('{}/{}_calib_labels.obj'.format(config.output_dir(),fltr),"rb")
    labels = pickle.load(filehandler)
    
    plt.rcParams.update({
        "font.family": "Sans", 
        "font.serif": ["DejaVu"],
        "figure.figsize":[20,10],
        "font.size": 14})
    period_to_mjd_range = config.observation_params()['periods']
    #    fig, axs = plt.subplots(2*len(period_to_mjd_range.keys()))
    fig = plt.figure()
    height_ratios = []
    for ii in period_to_mjd_range:
        height_ratios = height_ratios + [2,2]
    gs = gridspec.GridSpec(2*len(period_to_mjd_range.keys()), 1, height_ratios=height_ratios,hspace=0) 

    for i,period in enumerate(period_to_mjd_range):
        axs_idx = i*2
        ax0 = plt.subplot(gs[axs_idx])
        mjd_range = period_to_mjd_range[period]['mjd_range']
        for j in range(len(data)):
            mask = np.logical_and(data[j][:,0] >= mjd_range[0],data[j][:,0] <= mjd_range[1])
            mjd = data[j][mask,0]
            flux = data[j][mask,1]
            err = data[j][mask,2]
            ax0.errorbar(mjd, flux, yerr=err, ls='none', marker=".", label=str(scopes[j]))
        # filter and remove clipped datapoints from the calibrated lightcurve
        mask = np.logical_and(df[0] >= mjd_range[0],df[0] <= mjd_range[1])
        err_mask = np.logical_and(df[7] == False,mask)
        mjd_calib = df[0][mask].to_numpy()
        flux_calib = df[1][mask].to_numpy()
        err_calib = df[2][mask].to_numpy()
        mjd_calib_clipped = df[0][err_mask].to_numpy()
        flux_calib_clipped = df[1][err_mask].to_numpy()
        err_calib_clipped = df[2][err_mask].to_numpy()
        #ax0.plot(mjd_calib, flux_calib, color="black", label="Calibrated", alpha=0.5)
        #ax0.fill_between(mjd_calib, flux_calib+err_calib, flux_calib-err_calib, alpha=0.5, color="black")
        plt.setp(ax0.get_xticklabels(), visible=False)        
        ax1 = plt.subplot(gs[axs_idx+1], sharex = ax0)
        if mask_clipped:
            ax1.errorbar(mjd_calib_clipped, flux_calib_clipped, yerr=err_calib_clipped,
                         ls='none', marker=".",
                         color="black",
                         label="Sigma\nclipped\nvalues\nremoved\n(level={})".format(sig_level))
        else:
            ax1.errorbar(mjd_calib, flux_calib, yerr=err_calib,
                         ls='none', marker=".",
                         color="black",
                         label="Calibrated\nlight curve")
        ax1.set_ylabel('Flux (mJy) {}'.format(period))
        ax1.yaxis.set_label_coords(-0.04,1.35)
        # Shrink x=axis by 10% in order to fit the legend on the right
        box = ax0.get_position()
        ax0.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0+box.height*0.2, box.width * 0.9, box.height*0.8])
        if axs_idx == 0:
            ax0.set_title('{} Individual telescope and calibrated light curves for {}'.format(config.agn(),fltr))
            # Put a legend containing telescope names to the right in the cleared space
            ax0.legend(title="Telescope list",loc='center left', bbox_to_anchor=(1.0, 0.05))
            ax1.legend(loc='center left', bbox_to_anchor=(1.0, -1.2))
    ax1.set_xlabel('Time (days, MJD)')
    periods = [kk for kk in model.config().observation_params()['periods'].keys()]        
    output_file = '{}/{}_Calibration_Plot_{}_{}.pdf'.format(config.output_dir(),fltr,periods[0],periods[-1])
    if (os.path.exists(output_file) == True) and (overwrite == False):
        print('Not writing PyROA InterCalibratePlot, file exists: {}'.format(output_file))
    else:
        print('Writing calibration plot {}'.format(output_file))
        plt.savefig(output_file)
    if matplotlib.get_backend() == 'TkAgg':
        plt.show()
    else:
        plt.close()

    if corner_plot == False:
        return
    
    output_file = '{}/{}_{}_Calibration_CornerPlot.pdf'.format(config.output_dir(),select,fltr)
    if (os.path.exists(output_file) == True) and (overwrite == False):
        print('Not writing PyROA InterCalibratePlot (Corner), file exists: {}'.format(output_file))
        return

    # Generate params list
    samples_chunks = [np.transpose(samples_flat)[i:i + 3] for i in range(0, len(np.transpose(samples_flat)), 3)]
    params = []

    plt.rcParams.update({'font.size': 7})
    #Save Cornerplot to figure

    if (select == 'A') or (select == 'B') or (select == 'sig'):
        if select == 'A': shifter = 0
        if select == 'B': shifter = 1
        if select == 'sig': shifter = 2
        for i in range(len(data)):
            var = np.percentile(samples_chunks[i][shifter], [16, 50, 84])[1]
            params.append([var])

        params = list(chain.from_iterable(params))#Flatten into single array

        list_only = []
        for i in range(len(data)):
            list_only.append(i*3+shifter)

        print(list_only)
        print(np.array(labels)[list_only])
        fig = corner.corner(samples_flat[:,list_only], labels=np.array(labels)[list_only],
                            quantiles=[0.16, 0.5, 0.84], show_titles=True,
                            title_kwargs={"fontsize": 16}, truths=params);
    elif select == 'all':
        # A, B, sigma per scope
        for i in range(len(data)):
            A = np.percentile(samples_chunks[i][0], [16, 50, 84])[1]
            B = np.percentile(samples_chunks[i][1], [16, 50, 84])[1]
            sig = np.percentile(samples_chunks[i][2], [16, 50, 84])[1]
            params.append([A, B, sig])

        # Delta
        params.append([np.percentile(samples_chunks[-1], [16, 50, 84])[1]])
        params = list(chain.from_iterable(params))#Flatten into single array

        fig = corner.corner(samples_flat, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True,
                            title_kwargs={"fontsize": 16}, truths=params);

    else:
        print('Invalid chains select input ({}), no action'.format(select))
        return

    print('Writing calibration plot {}'.format(output_file))
    plt.savefig(output_file)

    if matplotlib.get_backend() == 'TkAgg':
        plt.show()
    else:
        plt.close()

def FitPlot(model,select_period,overwrite=False,noprint=True):

    config = model.config()
    roa_params = config.roa_params()
    ccf_params = config.ccf_params()

    add_var = roa_params["add_var"]
    sig_level = roa_params["sig_level"]
    delay_ref = roa_params["delay_ref"]
    roa_model = roa_params["model"]
    
    # We might chose to run the ROA for a single obervation period
    period_to_mjd_range = config.observation_params()['periods']
    if select_period not in period_to_mjd_range:
        raise Exception('Error: selected period {} not in observation periods for {}, check config'.format(select_period,config.agn()))
    mjd_range = config.observation_params()['periods'][select_period]['mjd_range']

    exclude_fltrs = roa_params["exclude_fltrs"]    
    fltrs = config.fltrs()

    fltrs = [fltr for fltr in fltrs if fltr not in exclude_fltrs]

    if len(fltrs) == 0:
        raise Exception('Insufficient filter bands passed to PyROA FitPlot: {} with reference filter {}'.format(fltrs,delay_ref))

    add_ext = '_{}_{}'.format(roa_params['model'],select_period)


    plt.rcParams.update({
        "font.family": "Sans",  
        "font.serif": ["DejaVu"],
        "figure.figsize":[16,12],
        "font.size": 12})  

    samples_file = '{}/samples_flat{}.obj'.format(config.output_dir(),add_ext)
    if Utils.check_file(samples_file) == False:
        input_ext = '_{}'.format(roa_model)
    else:
        input_ext = add_ext

    filehandler = open('{}/samples_flat{}.obj'.format(config.output_dir(),input_ext),"rb")
    samples_flat = pickle.load(filehandler)
        
#    filehandler = open('{}/samples{}.obj'.format(config.output_dir(),input_ext),"rb")
#    samples = pickle.load(filehandler)

#    filehandler = open('{}/labels{}.obj'.format(config.output_dir(),input_ext),"rb")
#    labels = pickle.load(filehandler)

    filehandler = open('{}/Lightcurves_models{}.obj'.format(config.output_dir(),input_ext),"rb")
    models = pickle.load(filehandler)
    
    #Split samples into chunks, 4 per lightcurve i.e A, B, tau, sig
    chunk_size = 4
    transpose_samples = np.transpose(samples_flat)
    #Insert zero where tau_0 would be 
    transpose_samples = np.insert(transpose_samples, [2], np.array([0.0]*len(transpose_samples[1])), axis=0)
    samples_chunks = [transpose_samples[i:i + chunk_size] for i in range(0, len(transpose_samples), chunk_size)] 

    fig = plt.figure(5)
    gs = fig.add_gridspec(len(fltrs), 1, hspace=0, wspace=0)
    band_colors=["royalblue", "darkcyan", "olivedrab", "maroon", "#ff6f00", "#ef0000", "#610000"]

    # get tau,mjd distribution extents
    tau_max = 0.0
    tau_min = 99999999.0
    mjd_max = 0.0
    mjd_min = 99999999.0
    
    # samples_chunks is length filters+1
    # and we skip the first (delay_ref) filter
    for i in range(len(fltrs)):
        sc = samples_chunks[i+1]
        tau_min = np.minimum(np.min(sc[2]), tau_min)
        tau_max = np.maximum(np.max(sc[2]), tau_max)
        
    data = []
    ccf_data = []
    #Loop over lightcurves
    for i,fltr in enumerate(fltrs):
        df_to_numpy = None
        calib_file = '{}/{}_{}.dat'.format(config.output_dir(),config.agn_name(),fltr)
        if Utils.check_file(calib_file,exit=True):
            # get mjd flux err from the calibration file as a numpy array of first three columns
            df = pd.read_csv(calib_file,
                             header=None,index_col=None,
                             quoting=csv.QUOTE_NONE,
                             delim_whitespace=True).sort_values(0)
            # Constrain to a single observation period if specified
            df = df[np.logical_and(df[0] >= mjd_range[0],
                                   df[0] <= mjd_range[1])]

            df = df[np.logical_and(df[0] >= mjd_range[0],
                                   df[0] <= mjd_range[1])]
            # Remove sigma clipped values
            df = df[df[7] == False].loc[:,0:2]
            # remove all points with more than 3 times the median error
            # to clean up the plot
            df = Utils.filter_large_sigma(df,sig_level,fltr,noprint=noprint)

            df_to_numpy = df.to_numpy()
            
        mjd_min = np.minimum(np.min(df_to_numpy[:,0]), mjd_min)
        mjd_max = np.maximum(np.max(df_to_numpy[:,0]), mjd_max)
        data.append(df_to_numpy)

        tlags_centroid = None
        if i > 0:
            centroidfile = '{}/Centroid_{}_{}_{}.dat'.format(config.output_dir(),select_period,delay_ref,fltr)
            if (Utils.check_file(centroidfile) == True):
                df = pd.read_csv(centroidfile,
                                 header=None,index_col=None,
                                 quoting=csv.QUOTE_NONE,
                                 delim_whitespace=True)
                tlags_centroid = df[0].to_numpy()
                tau_min = np.minimum(np.min(tlags_centroid), tau_min)
                tau_max = np.maximum(np.max(tlags_centroid), tau_max)
            else:
                print('No CCF data available for plot at {}'.format(centroidfile))
        ccf_data.append(tlags_centroid)

    tau_min = np.maximum(tau_min,-2.0)
    tau_max = np.minimum(tau_max,2.0)

    main_handles = []
    main_labels = []

    tau_handles = []
    tau_labels = []
    
    ilast = len(fltrs) - 1
    for i,fltr in enumerate(fltrs):        
        mjd = data[i][:,0]
        flux = data[i][:,1]
        err = data[i][:,2]

        # Add extra variance
        sig = np.percentile(samples_chunks[i+1][-1], 50)
        err = np.sqrt(err**2 + sig**2)

        # Organise subplot layout
        #ax = fig.add_subplot(gs[i])
        
        gssub = gs[i].subgridspec(2, 3, width_ratios=[5, 1, 1], height_ratios=[2,1], hspace=0, wspace=0)
        ax0 = fig.add_subplot(gssub[0,0])
        ax1 = fig.add_subplot(gssub[0,1])
        # Add the big legend far right location for later
        if i == 0:
            ax_legend = fig.add_subplot(gssub[0,2])
            ax_legend.axis('off')
        if i == int(float(len(fltrs))/2):
            ax_legend_tau = fig.add_subplot(gssub[0,2])
            ax_legend_tau.axis('off')            
        ax0_resid = fig.add_subplot(gssub[1,0])
        ax1_resid = fig.add_subplot(gssub[1,1])

        # Shrink x=axis by 10% in order to fit the legend on the right
        #box = ax0.get_position()
        #ax0.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        #box = ax1.get_position()
        #ax1.set_position([box.x1, box.y1, box.width * 0.8, box.height])
        #ax1.set_position([box.x0, box.y0+box.height*0.2, box.width * 0.9, box.height*0.8])
        
        # Plot Data
        ax0.errorbar(mjd, flux , yerr=err, ls='none', marker=".", color=band_colors[i], ms=2, elinewidth=0.75, label='{}'.format(fltr))
        # Plot Model
        t, m, errs = models[i+1]
        period_pick = np.logical_and(t >=mjd_min,t <= mjd_max)
        t = t[period_pick]
        m = m[period_pick]
        errs = errs[period_pick]
        ax0.plot(t,m, color="black", lw=1, label='ROA Model')
        ax0.fill_between(t, m+errs, m-errs, alpha=0.5, color="black")
        ax0.set_ylabel("Flux (mJy)",rotation=0,labelpad=40)
        ax0.text(0.025, 0.055, '{}'.format(fltr), fontsize = 14, transform = ax0.transAxes, color=band_colors[i])
        ax0.set_xlim(mjd_min,mjd_max)
        flux_margin = (np.max(flux) - np.min(flux))*0.15
        ax0.set_ylim(np.min(flux)-flux_margin,np.max(flux)+flux_margin)
        
        # calculate residuals 
        interp = interpolate.interp1d(t, m, kind="linear", fill_value="extrapolate")
        interpmodel = interp(mjd)
        residuals = interpmodel - flux
        # normalise residuals
        residual_mean = np.mean(residuals)
        residual_rms = np.std(residuals)
        residuals = (residuals - residual_mean)/residual_rms
        ax0_resid.plot(mjd,residuals, ls='none', marker='.', ms=0.75, color='#1f77b4',label='Residuals')
        ax0_resid.axhline(y = 0.0, color="black", ls="--",lw=0.5)
        #ax0_resid.set_ylabel("Residuals",rotation=0,labelpad=30)
        ax1_resid.hist(residuals, orientation="horizontal", color='#1f77b4')
        ax1_resid.axhline(y = 0.0, color="black", ls="--",lw=0.5)
        
        # Plot Time delay posterior distributions
        tau_samples = samples_chunks[i+1][2]
        roa_tau = np.percentile(tau_samples, [16, 50, 84])        
        dist_label = r'$\tau$'
        dist_label = dist_label + r'$_{'+fltrs[i]+r'}$ ROA dist'+'\n{:3.2f} (+{:3.2f},-{:3.2})'.format(roa_tau[1],
                                                                                                       roa_tau[0]-roa_tau[1],
                                                                                                       roa_tau[2]-roa_tau[1])
        ax1.hist(tau_samples, color=band_colors[i], bins=50, label=dist_label)
        ax1.axvline(x = roa_tau[1], color="black",lw=0.5)
        ax1.axvline(x = roa_tau[0] , color="black", ls="--",lw=0.5)
        ax1.axvline(x = roa_tau[2], color="black",ls="--",lw=0.5)
        ax1.axvline(x = 0, color="black",ls="--")    
        ax1.yaxis.set_tick_params(labelleft=False)
        ax1.set_xlim(tau_min,tau_max)
        
        if ccf_data[i] is not None:
            ax1.hist(ccf_data[i], bins = 50, color = 'grey', alpha=0.5, label=r'$\tau_{CCF}$ CCCD')
        
        if i == ilast:
            ax0_resid.set_xlabel("Time")
            ax0_resid.label_outer()
            handles, labels = ax0.get_legend_handles_labels()
            main_handles = main_handles + [handles[1],handles[0]]
            main_labels = main_labels + [labels[1],labels[0]]
            handles, labels = ax0_resid.get_legend_handles_labels()
            main_handles = main_handles + handles
            main_labels = main_labels + labels

            handles, labels = ax1.get_legend_handles_labels()
            tau_handles = tau_handles + handles
            tau_labels = tau_labels + labels
        else:
            plt.setp(ax0_resid.get_xticklabels(), visible=False)
            handles, labels = ax0.get_legend_handles_labels()
            main_handles = main_handles + [handles[1]]
            main_labels = main_labels + [labels[1]]

            handles, labels = ax1.get_legend_handles_labels()
            tau_handles = tau_handles + [handles[0]]
            tau_labels = tau_labels + [labels[0]]
            
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax1.get_yticklabels(), visible=False)
        plt.setp(ax1_resid.get_yticklabels(), visible=False)
        plt.setp(ax1_resid.get_xticklabels(), visible=False)
        ax1.set_yticks([])
        ax0_resid.set_yticks([])
        ax1_resid.set_yticks([])
        ax1_resid.set_xticks([])
        #ax0.legend(loc='lower left', fontsize=10)
        #ax0_resid.legend(loc='lower left', fontsize=10)

        #ax1.legend(loc='upper left', fontsize=10)
        
        if i == 0:
            title_ext = roa_model + ' {}'.format(select_period)
            ax0.set_title('{} Lightcurves {}'.format(config.agn(), title_ext), pad=10.0)

    # Put a legend containing the keys to the right in the cleared space
    ax_legend.legend(main_handles, main_labels, title='Calibrated Lightcurves',loc='center left', bbox_to_anchor=(0.0,-1))
    ax_legend_tau.legend(tau_handles, tau_labels, title='Delay Distributions',loc='center left', bbox_to_anchor=(0.0,-2))
    #ax_legend.legend(title="List 2",loc='center left', bbox_to_anchor=(0.0, -3.2))

    plt.subplots_adjust(wspace=0)

    output_file = '{}/ROA_LCs{}.pdf'.format(config.output_dir(),add_ext)
    if Utils.check_file(output_file) == True and overwrite==False:
        print('Not writing ROA FitPlot, file exists: {}'.format(output_file))
    else:
        print('Writing {}'.format(output_file))
        plt.savefig(output_file)
    if matplotlib.get_backend() == 'TkAgg':
        plt.show()
    else:
        plt.close()


def CalibrationOutlierPlot(model,select_period,fltr=None,add_model=False,overwrite=False,noprint=True,delta=None):

    config = model.config()
    fltrs = config.fltrs()
    fltr_ext = '_all'
    if fltr is not None:
        fltrs = [fltr]
        fltr_ext = '_{}'.format(fltr)
        
    period_to_mjd_range = config.observation_params()['periods']
    if select_period not in period_to_mjd_range:
        raise Exception('Error: selected period {} not in observation periods for {}, check config'.format(select_period,config.agn()))
    mjd_range = config.observation_params()['periods'][select_period]['mjd_range']
    
    # All the calibrated lightcurves pre-fitting on the same plot
    if add_model:
        calib_curve_plot = '{}/Calibrated_LCs_Model_{}{}.pdf'.format(config.output_dir(),select_period,fltr_ext)
    else:
        calib_curve_plot = '{}/Calibrated_LCs_{}{}.pdf'.format(config.output_dir(),select_period,fltr_ext)

    data=[]
    plt.style.use(['seaborn'])
    plt.rcParams.update({
        "font.family": "Sans",  
        "font.serif": ["DejaVu"],
        "figure.figsize":[18,7.5],
        "font.size": 14})
    if add_model is False:
            plt.rcParams.update({"figure.figsize":[14,7.5]})
    fig, axs = plt.subplots(len(fltrs),sharex=True)
    if add_model:
        #Add plots of normalised ROA data window weights
        height_ratios = []
        for i in range(len(fltrs)):
            height_ratios = height_ratios + [2,1]
        gs = gridspec.GridSpec(len(fltrs)*2, 1,height_ratios=height_ratios)
        range_step = 2
    else:
        gs = gridspec.GridSpec(len(fltrs), 1, hspace=0)
        range_step = 1

    remove_outliers = []
    if add_model:
        fig.suptitle('{} {} calibration analysis for filter band {}'.format(config.agn(),select_period,fltr))
    else:
        fig.suptitle('{} {} Calibrated light curves'.format(config.agn(),select_period))
    for i,ff in enumerate(fltrs):
        ff = fltrs[i]
        axsi = plt.subplot(gs[i*range_step])
        calib_file = '{}/{}_{}.dat'.format(config.output_dir(),config.agn_name(),ff)
        df_orig = pd.read_csv(calib_file,
                              header=None,index_col=None,
                              quoting=csv.QUOTE_NONE,
                              delim_whitespace=True).sort_values(0)
        # Constrain to a single observation year
        df = df_orig[np.logical_and(df_orig[0] >= mjd_range[0],
                                    df_orig[0] <= mjd_range[1])]

        if add_model:
            filehandler = open('{}/{}_calib_model.obj'.format(config.output_dir(),ff),"rb")
            load_model = pickle.load(filehandler)
            model_mask = np.logical_and(load_model[0] >= mjd_range[0],load_model[0] <= mjd_range[1])
            model_mjd = load_model[0][model_mask]
            model_flux = load_model[1][model_mask]
            model_err = load_model[2][model_mask]
        else:
            #for displaying all curves together, remove soft sigma clipped points
            #and any with particularly large error bars
            df = df[df[7] == False]
            df = Utils.filter_large_sigma(df,config.calibration_params()['sig_level'],ff,noprint=noprint)

        data.append(df)
        
        mjd = data[i][0]
        # X limits are three days either side of the calibrated datapoints
        axsi.set_xlim(np.min(mjd)-3,np.max(mjd)+3)
        axsi.set_xlim(np.min(mjd)-3,np.max(mjd)+3)
        flux = data[i][1]
        # Y limits are 5% either side of the calibrated flux
        axsi.set_ylim(np.min(flux)*0.95,np.max(flux)*1.05)
        err = data[i][2]
        if add_model:
            #Here we find influential outliers on the basis that some will crop up in periods with low point density
            #1) Get a blurred roa model using a 3 times Delta (if given) or 3x the median cadence window
            if delta is None:
                delta = Utils.median_cadence(mjd.to_numpy())
            blurred_roa_model = Utils.RunningOptimalAverage(mjd.to_numpy(),flux.to_numpy(),err.to_numpy(),delta*8.0)
            interp = interpolate.interp1d(blurred_roa_model[0], blurred_roa_model[1], kind="linear", fill_value="extrapolate")
            blurred_flux = interp(mjd)
            diff = np.abs(blurred_flux - flux)
            #Normalise the difference
            diff_mean = np.mean(diff)
            diff_rms = np.std(diff)
            diff = (diff - diff_mean)/diff_rms

            # two-sigma differences between the calibrated flux and the blurred ROA
            #potential_outlier = np.abs(diff) > 3.0
            potential_outlier = flux - blurred_flux > 0.25 * blurred_flux
            
            #Get point weight  density model based on delta (if given) or the median cadence
            density_model = Utils.WindowDensity(mjd.to_numpy(),err.to_numpy(),delta)
            # normalise the density model
            density_mean = np.mean(density_model[1])
            density_rms = np.std(density_model[1])
            density_model_norm = (density_model[1] - density_mean)/density_rms
            interp_density = interpolate.interp1d(density_model[0], density_model_norm, kind="linear", fill_value="extrapolate")
            density = interp_density(mjd)
            #potential outlier (a two-sigma difference from the blurred RunningOptimalAverage)
            #in relatively low density datapoint areas may by flagged for removal
            prm = np.logical_and(density < 0.5,potential_outlier)
            remove_outliers = np.array([mjd[prm],flux[prm]])
            if remove_outliers[0].shape[0] > 0:
                print('Found influential outliers with calibrated flux 25% away blurred ROA flux (delta=8*{}):'.format(delta))
                print(remove_outliers)
                #save *.dat aside with outliers removed
                model.remove_fltr_outliers(fltr,remove_outliers)
        axsi.errorbar(mjd, flux , yerr=err, ls='none', marker=".", ms=3.5, elinewidth=0.5,color="blue",label="Calibrated flux")
        if add_model:
            axsi.errorbar(mjd[prm], flux[prm], yerr=err[prm], ls='none', marker=".", ms=3.5, elinewidth=0.5,color="red",label="Outliers for removal")
            axsi.plot(blurred_roa_model[0],blurred_roa_model[1]*1.25,ls='dashed',color="red",label="ROA flux model (delta=8*{}) + 25%\n(under 25% outliers permitted)".format(delta))
            axsi.set_ylim(ymax=np.maximum(np.max(blurred_roa_model[1]*1.25)*1.05,np.max(flux)*1.05))
            axsi.set_ylabel('{} filter flux'.format(ff))
        else:
            axsi.set_ylabel('{}'.format(ff),color="blue")
            plt.setp(axsi.get_yticklabels(), visible=False)            
        if add_model:
            axsi.plot(model_mjd, model_flux, color="grey", label="ROA calibration flux model (delta={})".format(delta), alpha=0.5)
            axsi.fill_between(model_mjd, model_flux+model_err, model_flux-model_err, alpha=0.5, color="grey")
            axsi.legend()
            axsi = plt.subplot(gs[i*range_step+1])
            axsi.plot(density_model[0], density_model_norm, color="black", label="ROA window weights sum\n(normalised)",lw=0.5)
            axsi.axhline(y = 0.0, color="black", lw=0.5, ls="dashed")
            axsi.axhline(y = 0.5, lw=0.5, ls="dashed",color="red", label="ROA window weight = 0.5\n(over 0.5 outliers permitted)")
            axsi.legend()
    axsi.set_xlabel('Time (days, MJD)')
    if Utils.check_file(calib_curve_plot) == True and overwrite==False:
        print('Not writing Calibrated light curve plot, file exists: {}'.format(calib_curve_plot))
    else:
        plt.savefig(calib_curve_plot)
    if matplotlib.get_backend() == 'TkAgg':
        plt.show()
    else:
        plt.close()
                                   
    return

def CalibrationSNR(model,select_period,fltr=None,overwrite=False,noprint=True):
    print('Running PyROA CalibrationSNR')

    # references for convenience
    config = model.config()
    calib_params = config.calibration_params()
    roa_params = config.roa_params()

    delay_ref = roa_params["delay_ref"]
    sig_level = calib_params['sig_level']
    exclude_fltrs = roa_params["exclude_fltrs"]    
    fltrs = config.fltrs()

    fltrs = [ff for ff in fltrs if ff not in exclude_fltrs]

    if fltr is not None:
        fltrs = [fltr]    
    
    add_ext = '_{}'.format(roa_params['model'])
    
    data=[]

    # We might chose to display SNR for a single obervation period
    period_to_mjd_range = config.observation_params()['periods']
    if select_period not in period_to_mjd_range:
        raise Exception('Error: selected period {} not in observation periods for {}, check config'.format(select_period,config.agn()))
    mjd_range = config.observation_params()['periods'][select_period]['mjd_range']
    add_ext = add_ext + ' {}'.format(select_period)

    snr = []
    for ff in fltrs:
        calib_file = '{}/{}_{}.dat'.format(config.output_dir(),config.agn_name(),ff)
    
        if Utils.check_file(calib_file,exit=True):
            # get mjd flux err from the calibration file as a numpy array of first three columns
            df = pd.read_csv(calib_file,
                             header=None,index_col=None,
                             quoting=csv.QUOTE_NONE,
                             delim_whitespace=True).sort_values(0)
            df = df[np.logical_and(df[0] >= mjd_range[0],
                                   df[0] <= mjd_range[1])]
            # Remove large sigma outliers from the calibration model
            df = df[df[7] == False].loc[:,0:2]
            # remove all points with more than 3 times the median error
            df = Utils.filter_large_sigma(df,sig_level,ff,noprint=noprint)
            snr.append(Utils.signal_to_noise(df,sig_level,ff,noprint=noprint))

    ext = 'for AGN {} {}'.format(config.agn(),select_period)
    print('Signal to Noise ratio by filter {}'.format(ext))
    print(tabulate([snr],headers=fltrs))


def ConvergencePlot(model,select_period,overwrite=False):

    config = model.config()
    roa_params = config.roa_params()

    Nsamples = roa_params["Nsamples"]
    Nburnin = roa_params["Nburnin"]

    # tau initialisation by filter
    delay_ref = roa_params["delay_ref"]
    exclude_fltrs = roa_params["exclude_fltrs"]    
    fltrs = config.fltrs()
    
    fltrs = [fltr for fltr in fltrs if fltr not in exclude_fltrs]
    fltrs = [delay_ref] + fltrs

    add_ext = '_{}'.format(roa_params['model'])
    add_ext = add_ext + '_{}'.format(select_period)

    samples_file = '{}/samples{}.obj'.format(config.output_dir(),add_ext)
    if Utils.check_file(samples_file) == False:
        input_ext = '_{}'.format(roa_params['model'])
    else:
        input_ext = add_ext
    
    filehandler = open('{}/samples{}.obj'.format(config.output_dir(),input_ext),"rb")
    samples = pickle.load(filehandler)

    samples_flat = samples[14::15]
    ss = list(samples_flat.shape[1:])
    ss[0] = np.prod(samples_flat.shape[:2])
    # flatten the samples to a downsampled set
    samples = samples_flat.reshape(ss)
    
    init_chain_length=100

    # Compute the estimators for a few different chain lengths
    N = np.exp(np.linspace(np.log(init_chain_length), np.log(samples.shape[0]), 10)).astype(int)
    chain = samples.T
    gw2010 = np.empty(len(N))
    new = np.empty(len(N))
    for ii, nn in enumerate(N):
        gw2010[ii] = Utils.autocorr_gw2010(chain[:, :nn])
        new[ii] = Utils.autocorr_new(chain[:, :nn])

    fig = plt.figure(figsize=(8,6))
    # Plot the comparisons
    plt.loglog(N, gw2010, "o-", label="G&W 2010")
    plt.loglog(N, new, "o-", label="new")
    ylim = plt.gca().get_ylim()
    plt.plot(N, N / 50., "--k", label=r"$\tau = N/50$")
    plt.ylim(ylim)
    plt.xlabel("number of samples, $N$")
    plt.ylabel(r"$\tau$ estimates")
    plt.legend(fontsize=14)

    output_file = '{}/ROA_Convergence{}.pdf'.format(config.output_dir(),add_ext)
    if Utils.check_file(output_file) == True and overwrite==False:
        print('Not writing ROA ConvergencePlot, file exists: {}'.format(output_file))
    else:
        plt.savefig(output_file)
    if matplotlib.get_backend() == 'TkAgg':
        plt.show()
    else:
        plt.close()

def ChainsPlot(model,select_period,select='tau',start_sample=0,overwrite=False):
    config = model.config()
    roa_params = config.roa_params()

    delay_ref = roa_params["delay_ref"]
    roa_model = roa_params["model"]
    Nburnin = roa_params["Nburnin"]

    # We might chose to run the ROA for a single obervation period
    period_to_mjd_range = config.observation_params()['periods']
    if select_period not in period_to_mjd_range:
        raise Exception('Error: selected period {} not in observation periods for {}, check config'.format(select_period,config.agn()))
    mjd_range = config.observation_params()['periods'][select_period]['mjd_range']

    exclude_fltrs = roa_params["exclude_fltrs"]    
    fltrs = config.fltrs()

    fltrs = [fltr for fltr in fltrs if fltr not in exclude_fltrs]
    fltrs = [delay_ref] + fltrs

    if len(fltrs) == 0:
        raise Exception('Insufficient filter bands passed to PyROA FitPlot: {} with reference filter {}'.format(fltrs,delay_ref))
    
    add_ext = '_{}'.format(roa_model)
    add_ext = add_ext +  '_{}'.format(select_period)
    
    samples_file = '{}/samples{}.obj'.format(config.output_dir(),add_ext)
    if Utils.check_file(samples_file) == False:
        input_ext = '_{}'.format(roa_params['model'])
    else:
        input_ext = add_ext
    
    filehandler = open('{}/samples{}.obj'.format(config.output_dir(),input_ext),"rb")
    samples = pickle.load(filehandler)

    # flatten the big samples list
    samples_all_flat = samples[::15]
    ss = list(samples_all_flat.shape[1:])
    ss[0] = np.prod(samples_all_flat.shape[:2])
    samples_all_flat = samples_all_flat.reshape(ss)

    samples = samples_all_flat

    output_file = '{}/ROA_Chains{}_{}.pdf'.format(config.output_dir(),add_ext,select)
    if Utils.check_file(output_file) == True and overwrite==False:
        print('Not running ROA ChainsPlot, file exists: {}'.format(output_file))
        return
    
    # Plot each parameter
    labels = []
    for i in range(len(fltrs)):
        for j in ["A", "B",r"$\tau$", r"$\sigma$"]:
            labels.append(j+r'$_{'+fltrs[i]+r'}$')
    labels.append(r'$\Delta$')
    all_labels = labels.copy()
    del labels[2]
    print(labels)
    
    if type(select ) is int:
        ndim = select
        fig, axes = plt.subplots(ndim, figsize=(10, 2*ndim), sharex=True)
        #samples = sampler.get_chain()
        #labels = ["A", "B",r"$\tau$", r"$\sigma$"]
        ct = 0
        for i in range(start_sample,start_sample+ndim):
            ax = axes[ct]
            ax.plot(samples[:, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            #ax.set_ylabel("Param "+str(start_sample+i))
            #print(i,labels[i])
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
            ct += 1
        axes[-1].set_xlabel("Chain number")
    elif (select == 'all'):
        ndim = samples.shape[1]
        fig, axes = plt.subplots(ndim, figsize=(10, 2*ndim), sharex=True)
        #samples = sampler.get_chain()
        #labels = ["A", "B",r"$\tau$", r"$\sigma$"]
        ct = 0
        for i in range(ndim):
            ax = axes[ct]
            ax.plot(samples[:, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            #ax.set_ylabel("Param "+str(start_sample+i))
            #print(i,labels[i])
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
            ct += 1
            axes[-1].set_xlabel("Chain number")
    elif (select == 'tau') or (select == 'A') or (select == 'B') or (select == 'sig'):
        if select == 'A': shifter = 0
        if select == 'B': shifter = 1
        if select == 'tau': shifter = 2
        if select == 'sig': shifter = 3
        ndim = len(fltrs)
        fig, axes = plt.subplots(ndim-1, figsize=(10, 2*ndim), sharex=True)
        #samples = sampler.get_chain()
        #labels = ["A", "B",r"$\tau$", r"$\sigma$"]
        ct = 0
        mm = 0
        for i in range(ndim):
            if i != 0:
                if ndim == 2:
                    ax = axes
                else:
                    ax = axes[ct]
                ax.plot(samples[:, i*4+shifter+mm], "k", alpha=0.3)
                ax.set_xlim(0, len(samples))
                #ax.set_ylabel("Param "+str(start_sample+i))
                #print(i,all_labels[i*4+shifter])
                ax.set_ylabel(all_labels[i*4+shifter],fontsize=20)
                ax.yaxis.set_label_coords(-0.1, 0.5)
                ct+=1
            if i == 0:
                mm = -1
        if ndim == 2:
            axes.set_xlabel("Chain number")
        else:
            axes[-1].set_xlabel("Chain number")
    elif (select == 'delta'):
        fig, ax = plt.subplots(1, figsize=(10, 2))
        #samples = sampler.get_chain()
        #labels = ["A", "B",r"$\tau$", r"$\sigma$"]
        ax.plot(samples[:, -1], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        #ax.set_ylabel("Param "+str(start_sample+i))
        #print(i,all_labels[-1])
        ax.set_ylabel(all_labels[-1],fontsize=20)
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.set_xlabel("Chain number")
    else:
        print('Invalid chains select input ({}), no action'.format(select))
        return
	
    plt.savefig(output_file)
    if matplotlib.get_backend() == 'TkAgg':
        plt.show()
    else:
        plt.close()

def CornerPlot(model,select_period,select='tau',overwrite=False):
    config = model.config()
    roa_params = config.roa_params()

    delay_ref = roa_params['delay_ref']
    roa_model = roa_params['model']
    Nburnin = roa_params['Nburnin']

    exclude_fltrs = roa_params['exclude_fltrs']    
    fltrs = config.fltrs()

    fltrs = [fltr for fltr in fltrs if fltr not in exclude_fltrs]
    fltrs = [delay_ref] + fltrs

    if len(fltrs) == 0:
        raise Exception('Insufficient filter bands passed to PyROA CornerPlot: {} with reference filter {}'.format(fltrs,delay_ref))
    
    add_ext = '_{}'.format(roa_model)
    add_ext = add_ext + '_{}'.format(select_period)
    
    output_file = '{}/ROA_Corner{}_{}.pdf'.format(config.output_dir(),add_ext,select)
    if Utils.check_file(output_file) == True and overwrite==False:
        print('Not running ROA CornerPlot, file exists: {}'.format(output_file))
        return

    samples_file = '{}/samples_flat{}.obj'.format(config.output_dir(),add_ext)
    if Utils.check_file(samples_file) == False:
        input_ext = '_{}'.format(roa_params['model'])
    else:
        input_ext = add_ext

    filehandler = open('{}/samples_flat{}.obj'.format(config.output_dir(),input_ext),'rb')
    samples = pickle.load(filehandler)
    
    labels = []
    for i in range(len(fltrs)):
        for j in ['A', 'B',r'$\tau$', r'$\sigma$']:
            labels.append(j+r'$_{'+fltrs[i]+r'}$')
    labels.append(r'$\Delta$')
    all_labels = labels.copy()
    del labels[2]

    #print(labels)
    if (select == 'tau') or (select == 'A') or (select == 'B') or (select == 'sig'):
        if select == 'A': shifter = 0
        if select == 'B': shifter = 1
        if select == 'tau': shifter = 2
        if select == 'sig': shifter = 3

        list_only = []
        mm = 0
        for i in range(len(fltrs)):
            if i != 0:
                list_only.append(i*4+shifter+mm)
            if i == 0:
                mm = -1
        #print(list_only)
        #print(np.array(labels)[list_only])
        gg = corner.corner(samples[:,list_only],show_titles=True,
                           labels=np.array(labels)[list_only],
                           title_kwargs={'fontsize':19})
    elif select == 'all':
        gg = corner.corner(samples,show_titles=True,labels=labels)
    else:
        print('Invalid chains select input ({}), no action'.format(select))
        return
	
    plt.savefig(output_file)
    if matplotlib.get_backend() == 'TkAgg':
        plt.show()
    else:
        plt.close()


def FluxFlux(model,select_period,overwrite=False):
    """Flux-Flux analysis and Spectral energy distribution as
    estimated by PyROA."""

    # E(B-V) value of the line-of-sight extinction towards the AGN. 
    # The SED plot will be corrected by this amount following 
    # Fitzpatrick (1999) parametrisation.
    # Using 0.027 for this

    config = model.config()
    roa_params = config.roa_params()
    ccf_params = config.ccf_params()

    gal_ref = "u"    
    delay_ref = roa_params["delay_ref"]
    roa_model = roa_params["model"]
    mjd_range = config.observation_params()['periods'][select_period]['mjd_range']
    sig_level = roa_params["sig_level"]
    exclude_fltrs = roa_params["exclude_fltrs"]    
    fltrs = config.fltrs()

    fltrs = [fltr for fltr in fltrs if fltr not in exclude_fltrs]

    redshift = roa_params["redshift"]
    
    wavelengths = roa_params["wavelengths"]
    wavelengths = [wavelengths[fltr] for fltr in fltrs]

    band_colors = roa_params["band_colors"]
    band_colors = [band_colors[fltr] for fltr in fltrs]

    add_ext = '_{}_{}'.format(roa_params['model'],select_period)

    samples_file = '{}/samples_flat{}.obj'.format(config.output_dir(),add_ext)
    if Utils.check_file(samples_file) == False:
        input_ext = '_{}'.format(roa_model)
    else:
        input_ext = add_ext

    filehandler = open('{}/samples_flat{}.obj'.format(config.output_dir(),input_ext),"rb")
    samples_flat = pickle.load(filehandler)

    filehandler = open('{}/Lightcurves_models{}.obj'.format(config.output_dir(),input_ext),"rb")
    models = pickle.load(filehandler)

    filehandler = open('{}/X_t{}.obj'.format(config.output_dir(),input_ext),"rb")    
    norm_lc = pickle.load(filehandler)

    plt.rcParams.update({
        "font.family": "Serif",  
        "font.serif": ["Times New Roman"],
        "figure.figsize":[16,12],
        "font.size": 14})  

    ylab = r"F$_{\nu}$"+" / mJy"

    output_units = input_units = 'mJy'
    funits = 1*u.mJy
    ylab = r"F$_{\nu}$"+" / mJy"

    wave = np.array(wavelengths)

    #Split samples into chunks, 4 per lightcurve i.e A, B, tau, sig
    chunk_size=4
    transpose_samples = np.transpose(samples_flat)
    #Insert zero where tau_0 would be 
    transpose_samples = np.insert(transpose_samples, [2], np.array([0.0]*len(transpose_samples[1])), axis=0)
    samples_chunks = [transpose_samples[i:i + chunk_size] for i in range(0, len(transpose_samples), chunk_size)] 

    gal_spectrum,gal_spectrum_err,fnu_f,fnu_b,slope,slope_err = [],[],[],[],[],[]
    fnu_f_err,fnu_b_err = [], []
    
    fig = plt.figure(figsize=(10,7))
    xx = np.linspace(-15,5,300)
    max_flux = 0.0
    
    fac_flux = np.ones(len(wavelengths))
    for i in range(len(fltrs)):
        calib_file = '{}/{}_{}.dat'.format(config.output_dir(),config.agn_name(),fltrs[i])    
        df = pd.read_csv(calib_file,
                         header=None,index_col=None,
                         quoting=csv.QUOTE_NONE,
                         delim_whitespace=True).sort_values(0)
        df = df[np.logical_and(df[0] >= mjd_range[0],
                               df[0] <= mjd_range[1])]
        # Remove large sigma outliers from the calibration model
        data = df[df[7] == False].loc[:,0:2]
        # remove all points with more than 3 times the median error
        data = Utils.filter_large_sigma(data,sig_level,fltrs[i],noprint=True).to_numpy()

        # samples_chunks: add 1 to remove delay_ref data
        snu_mcmc = samples_chunks[i+1][0]
        cnu_mcmc = samples_chunks[i+1][1]            
        sig = np.percentile(samples_chunks[i+1][3], 50)
        
        mc_pl = np.zeros((200,xx.size))
        
        for lo in range(200):
            jj = int(np.random.uniform(0,snu_mcmc.size))
            mc_pl[lo] = cnu_mcmc[jj] + xx * snu_mcmc[jj]
            
        if fltrs[i] == gal_ref:
            x_gal_mcmc = -cnu_mcmc/snu_mcmc
            x_gal = np.median(x_gal_mcmc)
            x_gal_error = np.std(-cnu_mcmc/snu_mcmc)
                
        gal_spectrum_mcmc = np.median(cnu_mcmc) +  (x_gal_mcmc+x_gal_mcmc.std()) * np.median(snu_mcmc)
        
        gal_spectrum.append(gal_spectrum_mcmc.mean())
        gal_spectrum_err.append(gal_spectrum_mcmc.std())
        
        fnu_f_mcmc = snu_mcmc * (np.min(norm_lc[1]) - x_gal_mcmc)
        fnu_b_mcmc = snu_mcmc * (np.max(norm_lc[1]) - x_gal_mcmc)
        
        fnu_f.append(fnu_f_mcmc.mean())
        fnu_f_err.append(fnu_f_mcmc.std())
        
        fnu_b.append(fnu_b_mcmc.mean())
        fnu_b_err.append(fnu_b_mcmc.std())
        
        slope.append(np.median(snu_mcmc))
        slope_err.append(np.std(snu_mcmc))
        
        lin_fit = np.median(snu_mcmc) * xx + np.median(cnu_mcmc)
            
            
        if wavelengths != None:	   
                         
            if (input_units != 'flam') and (output_units !='flam'):
                wave = wavelengths[i] * u.Angstrom
                dd = funits
                #print(input_units,output_units)
                if output_units != 'fnu':
                    fac_flux[i] = dd.cgs.to(output_units).value
                else:
                    fac_flux[i] = dd.cgs.to('erg s^-1 cm^-2 Hz^-1').value

            if (input_units != 'flam') and (output_units =='flam'):
                wave = wavelengths[i] * u.Angstrom
                dd = funits/(wave**2)*ct.c

                #print(dd.cgs.to('erg/s/cm^2/Angstrom'))
                #print(funits.to('erg/s/cm**2/Hz'),wave,dd.cgs,fac_flux)
                fac_flux[i] = dd.cgs.to('erg s^-1 cm^-2 Angstrom^-1').value/1e-15

                #fac_flux[i] = dd.cgs.to('erg s^-1 cm^-2 Angstrom^-1').value
                #logo = int(np.log10(fnu_b[0]*fac_flux[0]))
                #print(fnu_b[i]*fac_flux[i],logo)
                #fac_flux[i]= fac_flux[i]*10**(logo)

            if (input_units == 'flam') and (output_units !='flam'):
                wave = wavelengths[i] * u.Angstrom
                dd = funits/ct.c*(wave**2)
                if output_units != 'fnu':
                    fac_flux[i] = dd.cgs.to(output_units).value
                else:
                    fac_flux[i] = dd.cgs.to('erg s^-1 cm^-2 Hz^-1').value

            #print(i,fac_flux)

        plt.fill_between(xx,(mc_pl.mean(axis=0)+mc_pl.std(axis=0))*fac_flux[i],
                         (mc_pl.mean(axis=0)-mc_pl.std(axis=0))*fac_flux[i],
                         color=band_colors[i],
                        alpha=0.3)
        interp_xt = np.interp(data[:,0],norm_lc[0],norm_lc[1])
        plt.errorbar(interp_xt,data[:,1]*fac_flux[i],
            	     yerr=np.sqrt(data[:,2]**2+sig**2)*fac_flux[i],
            	     color=band_colors[i],
                     ls='None',alpha=0.8)
        plt.plot(xx,lin_fit*fac_flux[i],color=band_colors[i],lw=3)
        max_flux = np.max([max_flux,np.max(data[:,1]*fac_flux[i])])

    fnu_f = np.array(fnu_f)
    fnu_f_err = np.array(fnu_f_err)
    fnu_b = np.array(fnu_b)
    fnu_b_err = np.array(fnu_b_err)
    slope = np.array(slope)
    slope_err = np.array(slope_err)
    gal_spectrum = np.array(gal_spectrum)
    gal_spectrum_err = np.array(gal_spectrum_err)

    plt.axvline(x=np.median(x_gal_mcmc+x_gal_mcmc.std()),color='r',
    			linestyle='-.',label=r'Galaxy')
    plt.axvline(x=np.min(norm_lc[1]),color='k',
    			linestyle='--',label=r'F$_{\rm faint}$')
    plt.axvline(x=np.max(norm_lc[1]),color='grey',
    			linestyle='--',label=r'F$_{\rm bright}$')

    lg = plt.legend(ncol=4)
    plt.xlim(x_gal-1,1+np.maximum(2,np.max(norm_lc[1])))
    #print()
    plt.ylim(-0.04*fac_flux[-1],max_flux*1.2)
    limits = None
    if limits != None: plt.ylim(-0.04,limits[1])
    
    plt.xlabel(r'$X_0 (t)$, Normalised driving light curve flux')
    plt.ylabel(ylab)
    plt.tight_layout()

    output_file = '{}/pyroa_fluxflux{}.pdf'.format(config.output_dir(),add_ext)
    if (os.path.exists(output_file) == True) and (overwrite == False):
        print('Not writing pyroa fluxflux plot, file exists: {}'.format(output_file))
    else:
        print('Writing pyroa fluxflux plot {}'.format(output_file))
        plt.savefig(output_file)
    if matplotlib.get_backend() == 'TkAgg':
        plt.show()
    else:
        plt.close()
    
    wave = np.array(wavelengths)
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111)
    xxx = np.arange(2000,9300)
    #plt.plot(xxx, 0.2*(xxx/3800)**(-7/3.)*(xxx**2/2.998e18)/1e-9*1000,'-',color='#6ab04c',
    #         label=r'F$_{\nu}\propto\lambda^{-1/3}$',lw=2)
    
    # AGN variability range
    plt.fill_between(wave/(1+redshift),(np.array(Utils.unred(wave,fnu_b,0.027)))*fac_flux,
        	     (np.array(Utils.unred(wave,fnu_f,0.027)))*fac_flux
                     ,color='k',alpha=0.1,label='AGN variability')
    ### F_bright - F_faint
    plt.errorbar(wave/(1+redshift),(np.array(Utils.unred(wave,fnu_b,0.027)) - \
                                    np.array(Utils.unred(wave,fnu_f,0.027)))*fac_flux,
                 yerr=np.sqrt((np.array(fnu_f_err))**2 + (np.array(fnu_b_err))**2)*fac_flux,
                 marker='.',linestyle='-',color='k',
                 label=r'F$_{\rm bright}$ - F$_{\rm faint}$',ms=15)

    ### AGN RMS
    plt.errorbar(wave/(1+redshift),np.array(Utils.unred(wave,slope,0.027))*fac_flux,
                 yerr=0,marker='o',linestyle='--',color='grey',label='AGN RMS')
        
    ### Galaxy spectrum
    gal_unred = Utils.unred(wave,gal_spectrum,0.027)*fac_flux
    plt.errorbar(wave/(1+redshift),gal_unred,
                 yerr=gal_spectrum_err*fac_flux,
                 marker='s',color='r',label='Galaxy',linestyle='-.')

    #print(fac_flux)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(np.min(wave/(1+redshift))-100,np.max(wave/(1+redshift))+100)
    #print(np.min(np.array(Utils.unred(wave,slope,0.027)))*0.7,max_flux*1.2)
    plt.ylim(np.minimum(np.min(gal_unred - gal_spectrum_err*fac_flux)*0.7,
                        np.min(np.array(Utils.unred(wave,slope,0.027)))*0.7*fac_flux[-1]),
             np.maximum(np.max(gal_unred),max_flux)*1.2)
    lg = plt.legend(ncol=2,loc='lower right')
    if redshift > 0:
        plt.xlabel(r'Rest Wavelength / $\mathrm{\AA}$')
    else:
        plt.xlabel(r'Observed Wavelength / $\mathrm{\AA}$')
    plt.ylabel(ylab)

    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
    ax.xaxis.set_minor_formatter(mtick.FormatStrFormatter('%.0f'))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(2000))
    plt.tight_layout()

    output_file = '{}/pyroa_SED{}.pdf'.format(config.output_dir(),add_ext)
    if (os.path.exists(output_file) == True) and (overwrite == False):
        print('Not writing pyroa SED plot, file exists: {}'.format(output_file))
    else:
        print('Writing pyroa SED plot {}'.format(output_file))
        plt.savefig(output_file)
    if matplotlib.get_backend() == 'TkAgg':
        plt.show()
    else:
        plt.close()

    d = {'wave': wave, 
    	'agn_b': np.array(fnu_b),
    	'agn_b_err': np.array(fnu_b_err),
    	'agn_f': np.array(fnu_f),
    	'agn_f_err': np.array(fnu_f_err),
    	'agn_rms': np.array(slope),
    	'agn_rms_err':np.array(slope_err),
    	'gal': np.array(gal_spectrum),
    	'gal_err': np.array(gal_spectrum_err),
    	'unred_agn_b': np.array(Utils.unred(wave,fnu_b,0.027)),
    	'unred_agn_b_err': np.array(Utils.unred(wave,fnu_b_err,0.027)),
    	'unred_agn_f': np.array(Utils.unred(wave,fnu_f,0.027)),
    	'unred_agn_f_err': np.array(Utils.unred(wave,fnu_f_err,0.027)),
    	'unred_agn_rms': np.array(Utils.unred(wave,slope,0.027)),
    	'unred_agn_rms_err':np.array(Utils.unred(wave,slope_err,0.027)),
    	'unred_gal': np.array(Utils.unred(wave,gal_spectrum,0.027)),
    	'unred_gal_err': np.array(Utils.unred(wave,gal_spectrum_err,0.027))}
    df = pd.DataFrame(data=d)

    output_file = '{}/pyroa_fluxflux{}.csv'.format(config.output_dir(),add_ext)
    if (os.path.exists(output_file) == True) and (overwrite == False):
        print('Not writing pyroa fluxflux data, file exists: {}'.format(output_file))
    else:
        print('Writing pyroa fluxflux data {}'.format(output_file))
        df.to_csv(output_file,index=False)

def LagSpectrum(model,select_period,overwrite=False):

    config = model.config()
    roa_params = config.roa_params()
    ccf_params = config.ccf_params()

    delay_ref = roa_params["delay_ref"]
    roa_model = roa_params["model"]
    mjd_range = config.observation_params()['periods'][select_period]['mjd_range']
    exclude_fltrs = roa_params["exclude_fltrs"]    
    fltrs = config.fltrs()
    # NEED TO CHANGE when delay_ref also in the ROA list
    #fltrs = [fltr for fltr in fltrs if fltr not in exclude_fltrs]
    fltrs = [fltr for fltr in fltrs if fltr not in exclude_fltrs]
    redshift = roa_params["redshift"]
    wavelengths = roa_params["wavelengths"]
    wavelengths = [wavelengths[fltr] for fltr in fltrs]
    band_colors = roa_params["band_colors"]
    band_colors = [band_colors[fltr] for fltr in fltrs]

    fltrs = [delay_ref] + fltrs
    
    add_ext = '_{}_{}'.format(roa_params['model'],select_period)

    samples_file = '{}/samples_flat{}.obj'.format(config.output_dir(),add_ext)
    if Utils.check_file(samples_file) == False:
        input_ext = '_{}'.format(roa_model)
    else:
        input_ext = add_ext

    filehandler = open('{}/samples_flat{}.obj'.format(config.output_dir(),input_ext),"rb")
    samples = pickle.load(filehandler)

    filehandler = open('{}/Lightcurves_models{}.obj'.format(config.output_dir(),input_ext),"rb")
    models = pickle.load(filehandler)

    filehandler = open('{}/X_t{}.obj'.format(config.output_dir(),input_ext),"rb")    
    norm_lc = pickle.load(filehandler)
    
    ss = np.where(np.array(fltrs) == delay_ref)[0][0]

    labels = []
    for i in range(len(fltrs)):
        for j in ["A","B",r"$\tau$",r"$\sigma$"]:
            labels.append(j+r'$_{'+fltrs[i]+r'}$')
    labels.append(r'$\Delta$')
    all_labels = labels.copy()
    del labels[ss*4+2]

    # To get ONLY lags
    shifter = 2

    list_only = []
    mm = 0
    ndim = len(fltrs)
    for i in range(ndim):
        if i != ss:
            list_only.append(i*4+shifter+mm)
        if i == ss:
            mm = -1
    # Get the 
    lag,lag_m,lag_p = np.zeros(ndim-1),np.zeros(ndim-1),np.zeros(ndim-1)
    for j,i in enumerate(list_only):
        #print(i)
        q50 = np.percentile(samples[:,i],50)
        q84 = np.percentile(samples[:,i],84)
        q16 = np.percentile(samples[:,i],16)
        lag[j] = q50
        lag_m[j] = q50-q16
        lag_p[j] = q84-q50
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111)

    plt.axhline(y=0,ls='--',alpha=0.5,)

    if band_colors == None: band_colors = 'k'*7

    mm = 0
    for i in range(lag.size):
        plt.errorbar(wavelengths[i]/(1+redshift),lag[i]/(1+redshift),
                     yerr=lag_m[i],marker='o',
                     color=band_colors[i])

    if redshift > 0:
        plt.xlabel(r'Rest Wavelength ($\mathrm{\AA}$)')
        plt.ylabel(r'$\tau_{\rm ROA}$ (day)')
    else:
        plt.xlabel(r'Observed Wavelength ($\mathrm{\AA}$)')
        plt.ylabel(r'$\tau$ (day)')

    plt.title('{} {} Lag Spectrum'.format(config.agn(),select_period))
    output_file = '{}/pyroa_lagspectrum{}.pdf'.format(config.output_dir(),add_ext)
    if (os.path.exists(output_file) == True) and (overwrite == False):
        print('Not writing pyroa lag spectrum plot, file exists: {}'.format(output_file))
    else:
        print('Writing pyroa lag spectrum plot {}'.format(output_file))
        plt.savefig(output_file)
    if matplotlib.get_backend() == 'TkAgg':
        plt.show()
    else:
        plt.close()
