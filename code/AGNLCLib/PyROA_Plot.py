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

    fltrs = [fltr for fltr in fltrs if fltr not in exclude_fltrs and fltr != delay_ref]
    fltrs = [delay_ref] + fltrs

    if len(fltrs) == 0:
        raise Exception('Insufficient filter bands passed to PyROA FitPlot: {} with reference filter {}'.format(fltrs,delay_ref))

    add_ext = '_{}_{}'.format(roa_params['model'],select_period)


    plt.rcParams.update({
        "font.family": "Sans",  
        "font.serif": ["DejaVu"],
        "figure.figsize":[16,12],
        "font.size": 14})  

    samples_file = '{}/samples_flat{}.obj'.format(config.output_dir(),add_ext)
    if Utils.check_file(samples_file) == False:
        input_ext = '_{}'.format(roa_model)
    else:
        input_ext = add_ext

    filehandler = open('{}/samples_flat{}.obj'.format(config.output_dir(),input_ext),"rb")
    samples_flat = pickle.load(filehandler)
        
    filehandler = open('{}/samples{}.obj'.format(config.output_dir(),input_ext),"rb")
    samples = pickle.load(filehandler)

    filehandler = open('{}/labels{}.obj'.format(config.output_dir(),input_ext),"rb")
    labels = pickle.load(filehandler)

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
    for i in range(len(fltrs)):
        sc = samples_chunks[i]
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

            # Remove large sigma residuals from the calibration model
            df = df[df[7] == False]

            df_to_numpy = df.loc[:,0:2].to_numpy()
            
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
        
    ilast = len(fltrs) - 1
    for i,fltr in enumerate(fltrs):        
        mjd = data[i][:,0]
        flux = data[i][:,1]
        err = data[i][:,2]

        # Add extra variance
        sig = np.percentile(samples_chunks[i][-1], 50)
        err = np.sqrt(err**2 + sig**2)

        # Organise subplot layout
        #ax = fig.add_subplot(gs[i])
        
        gssub = gs[i].subgridspec(2, 2, width_ratios=[5, 1], height_ratios=[2,1], hspace=0, wspace=0)
        ax0 = fig.add_subplot(gssub[0,0])
        ax1 = fig.add_subplot(gssub[0,1])
        ax0_resid = fig.add_subplot(gssub[1,0])
        ax1_resid = fig.add_subplot(gssub[1,1])

        # Plot Data
        ax0.errorbar(mjd, flux , yerr=err, ls='none', marker=".", color=band_colors[i], ms=2, elinewidth=0.75, label='Calibrated lightcurve')
        # Plot Model
        t, m, errs = models[i]
        period_pick = np.logical_and(t >=mjd_min,t <= mjd_max)
        t = t[period_pick]
        m = m[period_pick]
        errs = errs[period_pick]
        ax0.plot(t,m, color="black", lw=1, label='Model')
        ax0.fill_between(t, m+errs, m-errs, alpha=0.5, color="black")
        ax0.set_ylabel("Flux ({})".format(fltr),rotation=0,labelpad=30)
        ax0.set_xlim(mjd_min,mjd_max)
        flux_margin = (np.max(flux) - np.min(flux))*0.1
        ax0.set_ylim(np.min(flux)-flux_margin,np.max(flux)+flux_margin)
        
        # calculate residuals 
        interp = interpolate.interp1d(t, m, kind="linear", fill_value="extrapolate")
        interpmodel = interp(mjd)
        residuals = interpmodel - flux
        # normalise residuals
        residual_mean = np.mean(residuals)
        residual_rms = np.std(residuals)
        residuals = (residuals - residual_mean)/residual_rms
        ax0_resid.plot(mjd,residuals, ls='none', marker='.', ms=0.75, color='#1f77b4', label='Residual')
        ax0_resid.axhline(y = 0.0, color="black", ls="--",lw=0.5)
        ax1_resid.hist(residuals, orientation="horizontal", color='#1f77b4')
        ax1_resid.axhline(y = 0.0, color="black", ls="--",lw=0.5)
        
        # Plot Time delay posterior distributions
        tau_samples = samples_chunks[i][2]
        ax1.hist(tau_samples, color=band_colors[i], bins=50, label=r'$\tau$ ROA dist')
        ax1.axvline(x = np.percentile(tau_samples, [16, 50, 84])[1], color="black",lw=0.5)
        ax1.axvline(x = np.percentile(tau_samples, [16, 50, 84])[0] , color="black", ls="--",lw=0.5)
        ax1.axvline(x = np.percentile(tau_samples, [16, 50, 84])[2], color="black",ls="--",lw=0.5)
        ax1.axvline(x = 0, color="black",ls="--")    
        ax1.yaxis.set_tick_params(labelleft=False)
        ax1.set_xlim(tau_min,tau_max)
        
        if ccf_data[i] is not None:
            ax1.hist(ccf_data[i], bins = 50, color = 'grey', alpha=0.5, label=r'$\tau$ CCCD')
        
        if i == ilast:
            ax0_resid.set_xlabel("Time")
            ax0_resid.label_outer()
        else:
            plt.setp(ax0_resid.get_xticklabels(), visible=False)            
        
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax1.get_yticklabels(), visible=False)
        plt.setp(ax1_resid.get_yticklabels(), visible=False)
        plt.setp(ax1_resid.get_xticklabels(), visible=False)
        ax1.set_yticks([])
        ax1_resid.set_yticks([])
        ax1_resid.set_xticks([])
        ax0.legend(loc='lower left', fontsize=10)
        ax0_resid.legend(loc='lower left', fontsize=10)

        if i == 0:
            title_ext = roa_model + ' {}'.format(select_period)
            ax0.set_title('{} Lightcurves {}'.format(config.agn(), title_ext), pad=10.0)
        else:
            ax1.legend(loc='upper left', fontsize=10)
        

    plt.subplots_adjust(wspace=0)

    output_file = '{}/ROA_LCs{}.pdf'.format(config.output_dir(),add_ext)
    if Utils.check_file(output_file) == True and overwrite==False:
        print('Not running ROA FitPlot, file exists: {}'.format(output_file))
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
    fig, axs = plt.subplots(len(fltrs),sharex=True)
    if add_model:
        #Add plots of normalised ROA data window weights
        height_ratios = []
        for i in range(len(fltrs)):
            height_ratios = height_ratios + [2,1]
        gs = gridspec.GridSpec(len(fltrs)*2, 1,height_ratios=height_ratios)
        range_step = 2
    else:
        gs = gridspec.GridSpec(len(fltrs), 1)
        range_step = 1

    remove_outliers = []
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

        # Flag large sigma residuals from the calibration model for display
        sigma_clipped = df[7] == True
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
        axsi.legend()
        if add_model:
            axsi.plot(model_mjd, model_flux, color="grey", label="ROA calibration flux model (delta={})".format(delta), alpha=0.5)
            axsi.fill_between(model_mjd, model_flux+model_err, model_flux-model_err, alpha=0.5, color="grey")
        axsi.legend()
        if add_model:
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

    fltrs = [ff for ff in fltrs if ff not in exclude_fltrs and ff != delay_ref]    
    fltrs = [delay_ref] + fltrs
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
            df = Utils.filter_large_sigma(df,3.0,ff,noprint=noprint)
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
    
    fltrs = [fltr for fltr in fltrs if fltr not in exclude_fltrs and fltr != delay_ref]
    fltrs = [delay_ref] + fltrs

    add_ext = '_{}'.format(roa_params['model'])
    add_ext = add_ext + '_{}'.format(select_period)

    samples_file = '{}/samples_flat{}.obj'.format(config.output_dir(),add_ext)
    if Utils.check_file(samples_file) == False:
        input_ext = '_{}'.format(roa_params['model'])
    else:
        input_ext = add_ext
    
    filehandler = open('{}/samples_flat{}.obj'.format(config.output_dir(),input_ext),"rb")
    samples = pickle.load(filehandler)

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

    fltrs = [fltr for fltr in fltrs if fltr not in exclude_fltrs and fltr != delay_ref]
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

    fltrs = [fltr for fltr in fltrs if fltr not in exclude_fltrs and fltr != delay_ref]
    fltrs = [delay_ref] + fltrs

    if len(fltrs) == 0:
        raise Exception('Insufficient filter bands passed to PyROA FitPlot: {} with reference filter {}'.format(fltrs,delay_ref))
    
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

