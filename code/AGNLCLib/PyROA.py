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
import emcee
import pickle
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('xtick', labelsize=12) 
matplotlib.rc('ytick', labelsize=12) 
import scipy.interpolate as interpolate
import scipy.special as special
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import quad


def InterCalibrateFilt(model,fltr,overwrite=False):
    print('Running PyROA InterCalibrateFilt for filter {}'.format(fltr))

    # references for convenience
    config = model.config()
    calib_params = config.calibration_params()

    # set up scopes to be used for calibration
    scopes = config.scopes()
    exclude_scopes = calib_params.get("exclude_scopes",[])
    scopes = [scope for scope in scopes if scope not in exclude_scopes]
    print('Calibrating data for {} with {} excluded'.format(scopes,exclude_scopes))

    # set up the local variables
    data = []
    scopes_array = []
        
    for scope in scopes:
        scope_file = '{}/{}_{}_{}.dat'.format(config.output_dir(),config.agn_name(),fltr,scope)
        #Check if file is empty
        if os.stat(scope_file).st_size == 0:
            print("")
        else:
            dd = np.loadtxt(scope_file)
            data.append(dd)
            scopes_array.append([scope]*dd.shape[0])
            
    scopes_array = [item for sublist in scopes_array for item in sublist]

    output_file = '{}/{}_{}.dat'.format(config.output_dir(),config.agn_name(),fltr)

    if Utils.check_file(output_file) == True and overwrite==False:
        print('Not running filter {} calibration, file exists: {}'.format(fltr, output_file))
        return

    ########################################################################################    
    # No calibration data for this filter exists
    ########################################################################################        
    # Run MCMC to fit to data
    Npar = 3*len(data) + 1
    
    #Set inital conditions
    pos = [0]*(3*len(data) + 1)
    labels = [None]*(3*len(data) + 1)
    pos_chunks = [pos[i:i + 3] for i in range(0, len(pos), 3)]
    labels_chunks = [labels[i:i + 3] for i in range(0, len(labels), 3)]
    for i in range(len(data)):
        mjd = data[i][:,0]
        flux = data[i][:,1]
        err = data[i][:,2]
                
        pos_chunks[i][0] = pos_chunks[i][0] + 1.0 #Set intial A to one
        pos_chunks[i][1] = pos_chunks[i][1] + 0.0 #Set initial B to zero  
        pos_chunks[i][2] = np.mean(err)/5.0#2 #Set initial V to 1/5 of mean error
        
        labels_chunks[i][0] = "A"+str(i+1)
        labels_chunks[i][1] = "B"+str(i+1)        
        labels_chunks[i][2] = "\u03C3"+str(i+1)                
        
    pos_chunks[-1][0] = calib_params['init_delta']
    labels_chunks[-1][0] = "\u0394"
    #Store initial values for use in prior
    init_params_chunks = pos_chunks
        
    pos = np.array(list(chain.from_iterable(pos_chunks)))#Flatten into single array
    labels = list(chain.from_iterable(labels_chunks))#Flatten into single array     
    
    print("Initial Parameter Values")
    print(tabulate([pos.tolist()], headers=labels))

    #Define starting position
    pos = 1e-4 * np.random.randn(int(2.0*Npar), int(Npar)) + pos
    print("NWalkers="+str(int(2.0*Npar)))
    nwalkers, ndim = pos.shape
    sig_level = calib_params['sig_level']
    
    # 12 threads works if 12 virtual cores available
    # Reduce memory usage -> 6 threads 2023/06/14 as turgon has very little memory(?)
    with Pool(calib_params['Nparallel']) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, Utils.log_probability_calib, 
                                        args=(data, [calib_params['delta_prior'], calib_params['sigma_prior']],
                                              sig_level, init_params_chunks), pool=pool)
        sampler.run_mcmc(pos, calib_params['Nsamples'], progress=True);

    #Extract samples with burn-in of 10000 (default setting, see global.json)
    samples_flat = sampler.get_chain(discard=calib_params['Nburnin'], thin=15, flat=True)
                
    samples = sampler.get_chain()
                
    #####################################################################################
    # Repeat data shifting and ROA fit using best fit parameters

    #Split samples into chunks
    samples_chunks = [np.transpose(samples_flat)[i:i + 3] for i in range(0, len(np.transpose(samples_flat)), 3)]
    merged_mjd = []
    merged_flux = []
    merged_err = []
    A_values = []
    B_values = []
    avgs = []
    params = []

    for i in range(len(data)):
        mjd = data[i][:,0]
        flux = data[i][:,1]
        err = data[i][:,2]

        A = np.percentile(samples_chunks[i][0], [16, 50, 84])[1]
        A_values.append(A)
        B = np.percentile(samples_chunks[i][1], [16, 50, 84])[1]
        B_values.append(B)
        sig = np.percentile(samples_chunks[i][2], [16, 50, 84])[1]
        params.append([A, B, sig])
        #Shift data
        flux = (flux - B)/A
        #Add extra variance
        err = np.sqrt((err**2) + (sig**2))
        err = err/A

        avgs.append(np.average(flux, weights = 1.0/(err**2)))

        #Add shifted data to merged lightcurve
        for j in range(len(mjd)):
            merged_mjd.append(mjd[j])
            merged_flux.append(flux[j])
            merged_err.append(err[j])

    merged_mjd = np.array(merged_mjd)
    merged_flux = np.array(merged_flux)
    merged_err = np.array(merged_err)
    A_values = np.array(A_values)
    B_values = np.array(B_values)

    delta = np.percentile(samples_chunks[-1], [16, 50, 84])[1]
    params.append([delta])
    params = list(chain.from_iterable(params))#Flatten into single array

    #Calculate ROA to merged lc
    t, m, errs = Utils.RunningOptimalAverage(merged_mjd, merged_flux, merged_err, delta)
    
    Calibrated_mjd = []
    Calibrated_flux = []
    Calibrated_err = [] 
                
    Porc=Utils.CalculatePorc(merged_mjd, merged_flux, merged_err, delta)
    
    for i in range(len(data)):
        A = np.percentile(samples_chunks[i][0], [16, 50, 84])[1]
        B = np.percentile(samples_chunks[i][1], [16, 50, 84])[1]
        sig = np.percentile(samples_chunks[i][2], [16, 50, 84])[1]    
        
        #Originial lightcurves
        mjd = data[i][:,0]
        flux = data[i][:,1]
        err = data[i][:,2]
        #Add extra variance
        err = np.sqrt((err**2) + (sig**2))
     
        m_scaled = A*(m) + B
                
        #Model
        interp = interpolate.interp1d(t, m_scaled, kind="linear", fill_value="extrapolate")
        interpmodel = interp(mjd)
                
        #Sigma Clipping
        mask = (abs(interpmodel - flux) < sig_level*err)
        
        #Shift by parameters
        flux = (flux - B)/A          

        no_clipped = 0.0
        for j in range(len(mask)):
            if (mask[j]==False):
                no_clipped = no_clipped + 1
        print(no_clipped, "clipped, out of ", len(mjd), "data points")
        
        #Add shifted data to merged lightcurve        
        for j in range(len(mjd)):
            Calibrated_mjd.append(mjd[j])
            Calibrated_flux.append(flux[j])
            if (abs(interpmodel[j] - flux[j]) > sig_level*err[j]):
                Calibrated_err.append((abs(interpmodel[j] - flux[j])/sig_level))
            else:
                Calibrated_err.append(err[j])
                
    Calibrated_mjd = np.array(Calibrated_mjd)
    Calibrated_flux = np.array(Calibrated_flux)
    Calibrated_err = np.array(Calibrated_err)
                
    print("<A> = ", np.mean(A_values))
    print("<B> = ", np.mean(B_values))
    
    #Model
    interp = interpolate.interp1d(t, m, kind="linear", fill_value="extrapolate")
    interpmodel_j1 = interp(Calibrated_mjd)
    
    print(interpmodel_j1.shape)
    interp = interpolate.interp1d(t, errs, kind="linear", fill_value="extrapolate")
    error_j1 = interp(Calibrated_mjd)
    print(error_j1.shape)

    print(" >>>>> DELTA <<<<< ",delta)
        
    # Put all arrays in a pandas dataframe and export
    df = pd.DataFrame({
        'f1':Calibrated_mjd,
        'f2':Calibrated_flux,
        'f3':Calibrated_err,
        'str1':scopes_array,
        'f4':Porc,
        'f5':interpmodel_j1,
        'f6':error_j1
    }).sort_values('f1')
    df.to_csv(output_file,
              header=False,sep=' ',float_format='%25.15e',index=False,
              quoting=csv.QUOTE_NONE,escapechar=' ')

    # write out the calibration data as well
    filehandler = open('{}/{}_calib_samples_flat.obj'.format(config.output_dir(),fltr),"wb")
    pickle.dump(samples_flat,filehandler)
    filehandler = open('{}/{}_calib_samples.obj'.format(config.output_dir(),fltr),"wb")
    pickle.dump(samples,filehandler)
    filehandler = open('{}/{}_calib_labels.obj'.format(config.output_dir(),fltr),"wb")
    pickle.dump(labels,filehandler)

    
    return


########################################
# Diagnostic Graphs                    #
########################################
def InterCalibratePlot(model,fltr,overwrite=False):

    print('Running PyROA InterCalibratePlot for filter {}'.format(fltr))

    # references for convenience
    config = model.config()
    calib_params = config.calibration_params()

    # set up scopes to be used for calibration
    scopes = config.scopes()
    exclude_scopes = calib_params.get("exclude_scopes",[])
    scopes = [scope for scope in scopes if scope not in exclude_scopes]
    print('Calibrating data for {} with {} excluded'.format(scopes,exclude_scopes))

    # set up the local variables
    data = []
    scopes_array = []

    # read original LCs by scope
    for scope in scopes:
        scope_file = '{}/{}_{}_{}.dat'.format(config.output_dir(),config.agn_name(),fltr,scope)
        #Check if file is empty
        if os.stat(scope_file).st_size == 0:
            print("")
        else:
            data.append(np.loadtxt(scope_file))
            scopes_array.append([scope]*np.loadtxt(scope_file).shape[0])
            
    scopes_array = [item for sublist in scopes_array for item in sublist]
    
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

#    filehandler = open('{}/{}_calib_Lightcurves_models.obj'.format(config.output_dir(),fltr),"rb")
#    models = pickle.load(filehandler)
    
    output_file = '{}/{}_Calibration_Plot.pdf'.format(config.output_dir(),fltr)
    if (os.path.exists(output_file) == True) and (overwrite == False):
        print('Not running PyROA InterCalibratePlot, file exists: {}'.format(output_file))
        return

    plt.rcParams.update({
        "font.family": "Sans", 
        "font.serif": ["DejaVu"],
        "figure.figsize":[20,10],
        "font.size": 14})
        
    #Plot calibrated ontop of original lcs
    plt.title(str(fltr))
    #Plot data for filter
    for i in range(len(data)):
        mjd = data[i][:,0]
        flux = data[i][:,1]
        err = data[i][:,2]
        plt.errorbar(mjd, flux, yerr=err, ls='none', marker=".", label=str(scopes_array[i]), alpha=0.5)
                
    plt.errorbar(df[0], df[1], yerr=df[2], ls='none', marker=".", color="black", label="Calibrated")

    plt.xlabel("mjd")
    plt.ylabel("Flux")
    plt.legend()
    print('Writing calibration plot {}'.format(output_file))
    plt.savefig(output_file)
#    plt.close()
    plt.show()
    
    output_file = '{}/{}_Calibration_CornerPlot.pdf'.format(config.output_dir(),fltr)

    # Generate params list
    samples_chunks = [np.transpose(samples_flat)[i:i + 3] for i in range(0, len(np.transpose(samples_flat)), 3)]
    params = []

    # A, B, sigma per scope
    for i in range(len(data)):
        A = np.percentile(samples_chunks[i][0], [16, 50, 84])[1]
        B = np.percentile(samples_chunks[i][1], [16, 50, 84])[1]
        sig = np.percentile(samples_chunks[i][2], [16, 50, 84])[1]
        params.append([A, B, sig])

    # Delta
    params.append([np.percentile(samples_chunks[-1], [16, 50, 84])[1]])
    params = list(chain.from_iterable(params))#Flatten into single array
    
    plt.rcParams.update({'font.size': 7})
    #Save Cornerplot to figure
    fig = corner.corner(samples_flat, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True,
                        title_kwargs={"fontsize": 8}, truths=params);
    print('Writing calibration corner plot {}'.format(output_file))
    plt.savefig(output_file)
#    plt.close();
    plt.show()
    
    return

def Fit(model, overwrite=False, select_period=None):
    config = model.config()
    roa_params = config.roa_params()

    # Configure the ROA
    priors = [roa_params["A_prior"],
              roa_params["B_prior"],
              roa_params["tau_prior"],
              roa_params["delta_prior"],
              roa_params["v_prior"]]

    Nsamples = roa_params["Nsamples"]
    Nburnin = roa_params["Nburnin"]
    add_var = roa_params["add_var"]
    calc_P = roa_params["calc_P"]
    sig_level = roa_params["sig_level"]
    use_backend = roa_params["use_backend"]
    resume_progress = roa_params["resume_progress"]
    accretion_disk = roa_params["accretion_disk"]
    include_slow_comp = roa_params["include_slow_comp"]
    slow_comp_delta = roa_params["slow_comp_delta"]
    delay_dist = roa_params.get("delay_dist",False)
    mjd_range = None
    wavelengths = None
    
    # tau initialisation by filter
    init_tau_map = roa_params["init_tau"]
    delay_ref = roa_params["delay_ref"]
    init_delta = roa_params["init_delta"]
    exclude_fltrs = roa_params["exclude_fltrs"]    
    fltrs = config.fltrs()

    fltrs = [fltr for fltr in fltrs if fltr not in exclude_fltrs and fltr != delay_ref]
    init_tau = [init_tau_map[fltr] for fltr in fltrs]

    if len(fltrs) == 0:
        raise Exception('Insufficient filter bands passed to PyROA Fit: {} with reference filter {}'.format(fltrs,delay_ref))        

    # Arrays of LC filter bands to include in calculation, with reference filter at head
    fltrs.insert(0,delay_ref)
    init_tau.insert(0,0.0)
    psi_types_map = roa_params.get("psi_types",None)
    psi_types = None
    if psi_types_map:
        psi_types = [psi_types_map[fltr] for fltr in fltrs]

    add_ext = '_{}'.format(roa_params['model'])

    # We might chose to run the ROA for a single obervation period
    if select_period:
        period_to_mjd_range = config.observation_params()['periods']
        if select_period not in period_to_mjd_range:
            raise Exception('Error: selected period {} not in observation periods for {}, check config'.format(select_period,config.agn_name()))
        mjd_range = config.observation_params()['periods'][select_period]['mjd_range']
        add_ext = add_ext + '_{}'.format(select_period)

    output_file = '{}/samples_flat{}.obj'.format(config.output_dir(),add_ext)
    if Utils.check_file(output_file) == True and overwrite==False:
        print('Not running ROA Fit, file exists: {}'.format(output_file))
        return
    
    data = []
    for fltr in fltrs:
        calib_file = '{}/{}_{}.dat'.format(config.output_dir(),config.agn_name(),fltr)
    
        if Utils.check_file(calib_file,exit=True):
            # get mjd flux err from the calibration file as a numpy array of first three columns
            df_to_numpy = pd.read_csv(calib_file,
                                      header=None,index_col=None,
                                      quoting=csv.QUOTE_NONE,
                                      delim_whitespace=True).sort_values(0).loc[:,0:2].to_numpy()
            if mjd_range:
                df_to_numpy = df_to_numpy[np.logical_and(df_to_numpy[:,0] > mjd_range[0],
                                                         df_to_numpy[:,0] < mjd_range[1])]
            
            data.append(df_to_numpy)
    
    if (delay_dist==True):
        if (psi_types==None):
            psi_types = ["Gaussian"]*len(fltrs)
    else:
        psi_types = [None]*len(fltrs)
        
    if (init_tau == None):
        init_tau = [0]*len(data)
        if (delay_dist == True):
            init_tau = [1.0]*len(data)
        if (accretion_disk == True):
            init_tau = 5.0*(((np.array(wavelengths)/wavelengths[0]))**1.4)
            
    Nchunk = 2
    if (accretion_disk == False):
        Nchunk+=1
    if (add_var == True):
        Nchunk +=1
    if (delay_dist == True and accretion_disk == False):
        Nchunk+=1
        param_delete=2
    else:
        param_delete=1
        
    Npar =  Nchunk*len(data) + 1    
    if (accretion_disk==True):
        Npar =  Nchunk*len(data) + 3    
        param_delete=0
        delta = 0.0


    ###########################
    # Run MCMC to fit to data #
    ###########################
    
    #Choose intial conditions from mean and rms of data
    pos = [0]*Npar
    labels = [None]*Npar
    chunk_size = Nchunk
    assert(chunk_size == int((Npar - 1)/len(data)))
    
    pos_chunks = [pos[i:i + chunk_size] for i in range(0, len(pos), chunk_size)]
    labels_chunks = [labels[i:i + chunk_size] for i in range(0, len(labels), chunk_size)]
        
    size = 0
    merged_mjd = []
    merged_flux = []
    merged_err = []
    sizes = np.zeros(int(len(data)+1))
    for i in range(len(data)):
        mjd = data[i][:,0]
        flux = data[i][:,1]
        err = data[i][:,2]
        
        size = size + len(mjd)
        sizes[i+1] = len(mjd)   
     
        if (include_slow_comp==True):
            t_slow, m_slow, errs_slow = Utils.RunningOptimalAverage(mjd,flux,err, slow_comp_delta)
            m_slow = m_slow - np.mean(m_slow)
            m_s = interpolate.interp1d(t_slow, m_slow, kind="linear", fill_value="extrapolate")
            pos_chunks[i][0] = np.std(flux - m_s(mjd)) #Set intial A to rms of data
            pos_chunks[i][1] = np.mean(flux- m_s(mjd)) #Set initial B to mean of data

        else:        
            pos_chunks[i][0] = pos_chunks[i][0] + np.std(flux)# - m_s(mjd)) #Set intial A to rms of data
            pos_chunks[i][1] = np.mean(flux)#- m_s(mjd)) #Set initial B to mean of data
            
        if(add_var == True):
            pos_chunks[i][-1] =  np.mean(err)/5.0 #0.01 #Set initial V
            labels_chunks[i][-1] = "\u03C3"+str(i)
            
            
        if (delay_dist == True and accretion_disk == False):
            pos_chunks[i][3] = 1.0
            labels_chunks[i][3]="\u0394"+str(i)
       
                       
        labels_chunks[i][0] = "A"+str(i)
        labels_chunks[i][1] = "B"+str(i)
        if (accretion_disk == False):        
            labels_chunks[i][2] = "\u03C4" + str(i)
            pos_chunks[i][2] = init_tau[i]
        #Add shifted data to merged lightcurve        
        for j in range(len(mjd)):
            merged_mjd.append(mjd[j]-init_tau[i])
            merged_flux.append(flux[j])
            merged_err.append(err[j])
                
    merged_mjd = np.array(merged_mjd)
    merged_flux = np.array(merged_flux)
    merged_err = np.array(merged_err)
    
    P_slow=np.empty(len(data))
    #Calculate no. of parameters for a grid of deltas over the prior range
    if (calc_P == True):
        print("Calculating No. of parameters beforehand...")
        deltas=np.arange(priors[3][0], priors[3][1], 0.01)
        ps=np.empty(len(deltas))
        for i in tqdm(range(len(deltas))):

            ps[i]=CalculateP(merged_mjd, merged_flux, merged_err, deltas[i])            
        #P as a func of delta
        P_func=interpolate.interp1d(deltas, ps, kind="linear", fill_value="extrapolate")
    else:
        P_func=None
        
    slow_comps =[]                    
    if (include_slow_comp==True):
        for i in range(len(data)):
            t_sl, m_sl, errs_sl = Utils.RunningOptimalAverage(data[i][:,0], data[i][:,1], data[i][:,2], slow_comp_delta)            
            #params, pcov = scipy.optimize.curve_fit(Slow, data[i][:,0], data[i][:,1], p0=[4., -1.0, 59300] ,sigma=data[i][:,2], absolute_sigma=False)
            # perr = np.sqrt(np.diag(pcov))        
            t_sl=np.linspace(min(data[i][:,0]), max(data[i][:,0]), 1000)
            #m_sl = Slow(t_sl, params[0], params[1],params[2])
            #errs_sl = np.zeros(1000)
            #m_sl = Slow(np.linspace(59100, 59500, 1000), params[0], params[1],params[2]) - np.mean(m_sl)            
            slow_comps.append([t_sl, m_sl, errs_sl])
            P_slow[i] = CalculateP(data[i][:,0], data[i][:,1], data[i][:,2], slow_comp_delta)
            
    if (accretion_disk == True):
        pos_chunks[-1][0] = 1.0e4
        labels_chunks[-1][0] = "T1"
        pos_chunks[-1][1] = 0.75
        labels_chunks[-1][1] = "\u03B2"
        pos_chunks_table = pos_chunks
        labels_chunks_table = labels_chunks
        pos_chunks[-1][2] = init_delta#Initial delta
        labels_chunks[-1][2] = "\u0394"
        
        #Integral and interpolate
        Is=[]
        bs = np.linspace(0.34, 10.0, 5000)
        for i in range(len(bs)):
            Is.append(quad(integrand, 0, np.inf, args=(bs[i]))[0])
        integral= interpolate.interp1d(bs, Is, kind="linear", fill_value="extrapolate")
        Is=[]
        bs = np.linspace(0.34, 10.0, 5000)
        for i in range(len(bs)):
            Is.append(quad(integrand2, 0, np.inf, args=(bs[i]))[0])
        integral2= interpolate.interp1d(bs, Is, kind="linear", fill_value="extrapolate")

    else:
        # Initial delta
        pos_chunks[-1][0] = init_delta
        labels_chunks[-1][0] = "\u0394"
        integral=None
        integral2=None
        
    #Store initial values for use in prior
    init_params_chunks = pos_chunks

    # Flatten into single array
    pos = list(chain.from_iterable(pos_chunks))
    labels = list(chain.from_iterable(labels_chunks))
    
    pos_rem = 2
    if (accretion_disk == False):
        pos = np.delete(pos, pos_rem) 
        labels = np.delete(labels, pos_rem)
    
    if (delay_dist==True and accretion_disk == False):
        pos = np.delete(pos, [2]) 
        labels = np.delete(labels, [2])

    print("Initial Parameter Values")

    print(tabulate([pos.tolist()], headers=labels))
    
    #Define starting position    
    pos = 0.2*pos* np.random.randn(int(2.0*Npar), int(Npar - param_delete)) + pos
    nwalkers, ndim = pos.shape
    print('NWalkers={}'.format(nwalkers))

    # Make sure initial positions aren't outside priors
    # priors_upper=[]
    # priors_lower=[]
    # for i in range(len(priors)):
    #     priors_upper.append(priors[i][0])
    #     priors_lower.append(priors[i][1])

    # for i in range(nwalkers):
    #     pos[i,:][ pos[i,:] > priors_upper] = priors_upper[pos[i,:] > priors_upper]
    #     pos[i,:][ pos[i,:] < priors_lower] = priors_lower[pos[i,:] < priors_lower]
    
    #Backend
    if (use_backend == True):
        filename = "Fit.h5"
        backend = emcee.backends.HDFBackend(filename)
        if (resume_progress == True):
            print("Backend size: {0}".format(backend.iteration))
            pos = None
    else:
        backend = None

    # 12 threads works if 12 virtual cores available
    # Reduce memory usage -> 8 threads 2023/06/18

    samples_file = '{}/samples{}.obj'.format(config.output_dir(),add_ext)
    if Utils.check_file(samples_file) == True:
        print('Samples file exists- not running MCMC: {}'.format(samples_file))
        filehandler = open(samples_file,"rb")
        samples = pickle.load(filehandler)
        if samples.shape[0] != Nsamples:
            raise Exception('Tried to load samples from {} as the file exists but Nsamples={} and samples file length is {}'.format(samples_file,Nsamples,samples.shape[0]))
        # Duplicate the logic from the emcee chains sampler below to contruct samples_flat
        #v = getattr(self, name)[discard + thin - 1 : self.iteration : thin]
        #if flat:
        #    s = list(v.shape[1:])
        #    s[0] = np.prod(v.shape[:2])
        #    return v.reshape(s)
        #return v
        samples_flat = samples[Nburnin + 14 ::15]
        ss = list(samples_flat.shape[1:])
        ss[0] = np.prod(samples_flat.shape[:2])
        samples_flat = samples_flat.reshape(ss)
    else:    
        with Pool(roa_params['Nparallel']) as pool:

            sampler = emcee.EnsembleSampler(nwalkers, ndim, Utils.log_probability, args=[data, priors, add_var, size,sig_level, include_slow_comp, slow_comp_delta, P_func, slow_comps, P_slow, init_delta, delay_dist, psi_types, accretion_disk, wavelengths, integral, integral2, init_params_chunks], pool=pool, backend=backend)
            sampler.run_mcmc(pos, Nsamples, progress=True);

            # Extract samples with some (configured) burnin, usually Nburnin=10000
            samples_flat = sampler.get_chain(discard=Nburnin, thin=15, flat=True)
    
            samples = sampler.get_chain()

    ##############################################################
    # Repeat data shifting and ROA fit using best fit parameters #
    ##############################################################
    
    transpose_samples = np.transpose(samples_flat)

    if (delay_dist==True and accretion_disk == False):
        transpose_samples=np.insert(transpose_samples, [2], np.array([0.0]*len(transpose_samples[1])), axis=0)              #Insert zero for reference delay dist

    if (accretion_disk == False):
        transpose_samples= np.insert(transpose_samples, pos_rem, np.array([0.0]*len(transpose_samples[1])), axis=0)    #Insert zero for reference delay          
                     
    # Split samples into chunks
    samples_chunks = [transpose_samples[i:i + chunk_size] for i in range(0, len(transpose_samples), chunk_size)] 
    
    # Extract delta and extra variance parameters as last in params list
    if (accretion_disk == False):
        delta = np.percentile(samples_chunks[-1][0], [16, 50, 84])[1]
        wavelengths = [None]*len(data)
        T1 = None 
        b = None
        integral = None
        
    if (delay_dist == True and accretion_disk == False):
        delta = np.percentile(samples_chunks[-1][0], [16, 50, 84])[1]
        rmss = np.zeros(size)
        taus = np.zeros(size)
    if (accretion_disk == True):
        rmss = np.zeros(size)
        taus = np.zeros(size)
        T1 = np.percentile(samples_chunks[-1][0], [16, 50, 84])[1]
        b = np.percentile(samples_chunks[-1][1], [16, 50, 84])[1]
        delta = np.percentile(samples_chunks[-1][2], [16, 50, 84])[1]

    # Loop through each lightcurve and shift data by parameters
    merged_mjd = np.zeros(size)
    merged_flux = np.zeros(size)
    merged_err = np.zeros(size)

    params=[]
    avgs = []    
    slow_comps_out = []
    prev=0
    for i in range(len(data)):
        mjd = data[i][:,0]
        flux = data[i][:,1]
        err = data[i][:,2]

        A = np.percentile(samples_chunks[i][0], [16, 50, 84])[1]
        B = np.percentile(samples_chunks[i][1], [16, 50, 84])[1]
        if (accretion_disk == False):
            tau = np.percentile(samples_chunks[i][2], [16, 50, 84])[1]
            params.append([A, B, tau])    
        else:
            l_0 = wavelengths[0]
            l = wavelengths[i] - l_0        
            l_delay_ref = wavelengths[0] - l_0
            tau_0 = (l_0*1e-10*1.3806e-23*T1/(6.63e-34*3e8))**(1.0/b)
            # Measure mean from delay reference
            tau = tau_0*((l/l_0)**(1.0/b))*8.0*(np.pi**4)/(15.0*integral(b)) - tau_0*((l_delay_ref/l_0)**(1.0/b))*8.0*(np.pi**4)/(15.0*integral(b))
            tau_rms = np.sqrt((tau_0**2)*((l/l_0)**(2.0/b))*integral2(b)/integral(b))
            params.append([A, B])    
        #Print delays
        
        if (delay_dist == True and accretion_disk == False):
            smpls = samples_chunks[i][2]
            if (psi_types[i] == "TruncGaussian"):
                smpls = peaktomean(samples_chunks[i][2], samples_chunks[0][2], samples_chunks[i][3])
            mean_delay = np.percentile(smpls, [16, 50, 84])
            if (i != 0):
                print("Filter: " + str(fltrs[i]))
                print('Mean Delay, error: %10.5f  (+%10.5f -%10.5f)'%(mean_delay[1], mean_delay[2] - mean_delay[1], mean_delay[1] - mean_delay[0]))
            else:
                print("Filter: " + str(fltrs[i]))
                print("Mean Delay, error: 0.00 (fixed)")
        elif(delay_dist == False and accretion_disk == False):
            delay = np.percentile(samples_chunks[i][2], [16, 50, 84])
            if (i != 0):
                print("Filter: " + str(fltrs[i]))
                print('Delay, error: %10.5f  (+%10.5f -%10.5f)'%(delay[1], delay[2] - delay[1], delay[1] - delay[0]))  
            else:
                print("Filter: " + str(fltrs[i]))
                print("Delay, error: 0.00 (fixed)")
                
        if (accretion_disk == True):
            tau_0_samples =  (l_0*1e-10*1.3806e-23*samples_chunks[-1][0]/(6.63e-34*3e8))**(1.0/samples_chunks[-1][1])
            tau_samples = tau_0_samples*((l/l_0)**(1.0/samples_chunks[-1][1]))*8.0*(np.pi**4)/(15.0*integral(samples_chunks[-1][1])) - tau_0_samples*((l_delay_ref/l_0)**(1.0/samples_chunks[-1][1]))*8.0*(np.pi**4)/(15.0*integral(samples_chunks[-1][1]))
            mean_delay = np.percentile(tau_samples, [16, 50, 84])
            if (i != 0):
                print("Filter: " + str(fltrs[i]))
                print('Mean Delay, error: %10.5f  (+%10.5f -%10.5f)'%(mean_delay[1], mean_delay[2] - mean_delay[1], mean_delay[1] - mean_delay[0]))
            else:
                print("Filter: " + str(fltrs[i]))
                print("Mean Delay, error: 0.00 (fixed)")
                            
            
        if (delay_dist == True and accretion_disk == False):
            #if (PowerLaw == False):
            tau_rms=0.0
            if (i>0):
                tau_rms = np.percentile(samples_chunks[i][3], [16, 50, 84])[1]
           # else:
               # tau_rms =  np.percentile(samples_chunks[-1][1], [16, 50, 84])[1]*(((wavelengths[i]/wavelengths[0]) - 0.999)** np.percentile(samples_chunks[-1][2], [16, 50, 84])[1])               
            params.append([tau_rms])
            
        if (add_var == True):
            V =  np.percentile(samples_chunks[i][-1], [16, 50, 84])[1]
            err = np.sqrt((err**2) + (V**2))
            params.append([V])                                       


        if (include_slow_comp==True):
            t_slow, m_slow, errs_slow = slow_comps[i]
            m_s = interpolate.interp1d(t_slow, m_slow, kind="linear", fill_value="extrapolate")
            slow_comps_out.append(m_s)
            flux = (flux - B - m_s(mjd))/A
            
        else:
            flux = (flux - B )/A          
        #Shift data
        mjd = mjd - tau
        err = err/A

        for j in range(len(mjd)):
            merged_mjd[int(j+ prev)] = mjd[j]
            merged_flux[int(j+ prev)] = flux[j]
            merged_err[int(j+ prev)] = err[j]
            if (delay_dist == True or accretion_disk == True):
                rmss[int(j+ prev)] = tau_rms
                taus[int(j+ prev)] = tau                
                #factors[int(j+ prev)] = delta/np.sqrt(delta**2 + (tau_rms)**2)#/delta
     
        prev = int(prev + len(mjd))
    
    if (accretion_disk == False):
        params.append([delta])
    else:
        params.append([T1, b])

            
    params = list(chain.from_iterable(params))#Flatten into single array
    
    
    if (accretion_disk == False):  
        params=np.delete(params, pos_rem)   
    
    if (delay_dist==True and accretion_disk == False):
        params=np.delete(params, [2])
    #Calculate ROA to merged lc

    if (delay_dist == False and accretion_disk == False):
    

        t, m, errs = Utils.RunningOptimalAverage(merged_mjd, merged_flux, merged_err, delta)
    
        #Normalise lightcurve
        m_mean = np.mean(m)#np.average(m, weights = 1.0/(errs**2))
        m_rms = np.std(m)


        m = (m-m_mean)/m_rms
        errs = errs/m_rms
        
    else:
       # ws = CalcWind(merged_mjd, delta, rmss)
        factors, conv, x, d = Utils.CalcWinds(merged_mjd, merged_flux, merged_err, delta, rmss, len(data), sizes,  taus, psi_types, wavelengths, T1, b, integral)
        t,m_all,errs_all, P_all = Utils.RunningOptimalAverageConv(merged_mjd, merged_flux, merged_err, d, factors, conv, x)     

        #t,m_all,errs_all = Utils.RunningOptimalAverage3(merged_mjd, merged_flux, merged_err, delta, rmss, ws)
        #Calculate Norm. conditions 

       # m_mean = np.mean(m)
       # m_rms = np.std(m)
       # m = (m-m_mean)/m_rms
        #errs = errs/m_rms

    #Output model for specific lightcurves    
    models=[]
    prev=0
    for i in range(len(data)):
        mjd = data[i][:,0]
        flux = data[i][:,1]
        err = data[i][:,2]
        
        A = np.percentile(samples_chunks[i][0], [16, 50, 84])[1]
        B = np.percentile(samples_chunks[i][1], [16, 50, 84])[1]
        if (accretion_disk == False):
            tau = np.percentile(samples_chunks[i][2], [16, 50, 84])[1]       
        else:
            l_0 = wavelengths[0]
            l = wavelengths[i] - l_0        
            l_delay_ref = wavelengths[0] - l_0
            tau_0 = (l_0*1e-10*1.3806e-23*T1/(6.63e-34*3e8))**(1.0/b)
            tau = tau_0*((l/l_0)**(1.0/b))*8.0*(np.pi**4)/(15.0*integral(b)) - tau_0*((l_delay_ref/l_0)**(1.0/b))*8.0*(np.pi**4)/(15.0*integral(b)) #Measure mean from delay reference
            tau_rms = np.sqrt((tau_0**2)*((l/l_0)**(2.0/b))*integral2(b)/integral(b))
        

        if (delay_dist == False and accretion_disk == False):
            t_shifted = t + tau
            #interp = interpolate.interp1d(t_shifted, m, kind="linear", fill_value="extrapolate")
            m_m = m#interp(t)
                    
            if (include_slow_comp==True):
                m_s = np.interp(t_shifted, slow_comps[i][0], slow_comps[i][1])
                m_scaled = A*(m_m) + B + m_s
            else:
                m_scaled = A*(m_m) + B
         
            #Model
            # interp = interpolate.interp1d(t, m_scaled, kind="linear", fill_value="extrapolate")
            #model = interp(t)
            #model errors
            model_errs = errs*A
            models.append([t_shifted, m_scaled, model_errs])

        else:
            if (accretion_disk == False):
                if (i>0):
                    tau_rms = np.percentile(samples_chunks[i][3], [16, 50, 84])[1]
                else:
                    tau_rms=0.0
           # else:
                #tau_rms =  np.percentile(samples_chunks[-1][1], [16, 50, 84])[1]*(((wavelengths[i]/wavelengths[0]) - 0.999)** np.percentile(samples_chunks[-1][2], [16, 50, 84])[1])      
                
            delta_new = np.max(d)
            
            mx=max(merged_mjd)
            mn=min(merged_mjd)
            length = abs(mx-mn)
            t = np.arange(mn, mx, length/(1000)) 

            
            ts, Xs, errss = Utils.RunningOptimalAverageOutConv(t, merged_mjd, merged_flux, merged_err, factors, conv, prev, x, delta_new)            
                        
            m_mean = np.mean(m_all[prev : int(prev + len(mjd))])
            m_rms = np.std(m_all[prev : int(prev + len(mjd))])


            Xs = (Xs-m_mean)/m_rms
            errss = errss/m_rms
            
            if (include_slow_comp==True):
                t_shifted = t + tau
                m_s = np.interp(t_shifted, slow_comps[i][0], slow_comps[i][1])
                
                model = A*Xs + B + m_s
            else:
                model = A*Xs + B
        
            model_errs = errss*A
            models.append([t+tau, model, model_errs])
            
            if (i ==0):         
                t,m,errs = [t+tau, Xs,errss]
                       
        prev = int(prev + len(mjd))
        
    print('')
    print('')
    print('Best Fit Parameters')
    print(tabulate([params], headers=labels))
    
    #Write samples to file
    filehandler = open('{}/samples_flat{}.obj'.format(config.output_dir(),add_ext),"wb")
    pickle.dump(samples_flat,filehandler)
    filehandler = open('{}/samples{}.obj'.format(config.output_dir(),add_ext),"wb")
    pickle.dump(samples,filehandler)

    filehandler = open('{}/labels{}.obj'.format(config.output_dir(),add_ext),"wb")
    pickle.dump(labels,filehandler)

    filehandler = open('{}/X_t{}.obj'.format(config.output_dir(),add_ext),"wb")
    pickle.dump([t, m, errs],filehandler)
    if (include_slow_comp==True):      
        filehandler = open('{}/Slow_comps{}.obj'.format(config.output_dir(),add_ext),"wb")
        pickle.dump(slow_comps_out,filehandler)
        
    filehandler = open('{}/Lightcurves_models{}.obj'.format(config.output_dir(),add_ext),"wb")
    pickle.dump(models,filehandler)
    
    return


########################################
# Diagnostic Graphs                    #
########################################
def FitPlot(model,select_period,overwrite=False):

    config = model.config()
    roa_params = config.roa_params()
    ccf_params = config.ccf_params()

    add_var = roa_params["add_var"]
    sig_level = roa_params["sig_level"]
    delay_ref = roa_params["delay_ref"]
    roa_model = roa_params["model"]
    mjd_range = None

    ccf_flux_jump_sig_level = ccf_params.get('flux_jump_sig_level',None)
    ccf_sig_level = ccf_params['sig_level']
    
    # We might chose to run the ROA for a single obervation period
    period_to_mjd_range = config.observation_params()['periods']
    if select_period not in period_to_mjd_range:
        raise Exception('Error: selected period {} not in observation periods for {}, check config'.format(select_period,config.agn_name()))
    mjd_range = config.observation_params()['periods'][select_period]['mjd_range']

    exclude_fltrs = roa_params["exclude_fltrs"]    
    fltrs = config.fltrs()

    fltrs = [fltr for fltr in fltrs if fltr not in exclude_fltrs and fltr != delay_ref]
    fltrs = [delay_ref] + fltrs

    if len(fltrs) == 0:
        raise Exception('Insufficient filter bands passed to PyROA FitPlot: {} with reference filter {}'.format(fltrs,delay_ref))

    add_ext = '_{}_{}'.format(roa_params['model'],select_period)

    output_file = '{}/ROA_LCs{}.pdf'.format(config.output_dir(),add_ext)
    if Utils.check_file(output_file) == True and overwrite==False:
        print('Not running ROA FitPlot, file exists: {}'.format(output_file))
        return
    
    plt.rcParams.update({
        "font.family": "Sans",  
        "font.serif": ["DejaVu"],
        "figure.figsize":[40,30],
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
                             delim_whitespace=True).sort_values(0).loc[:,0:2]
            # filter datapoints with large error
            df = Utils.filter_large_sigma(df,ccf_sig_level,fltr)
        
            # filter datapoints with large flux jumps either side
#            if ccf_flux_jump_sig_level:
#                df = Utils.filter_large_sigma_jumps(df,ccf_flux_jump_sig_level,fltr)
            
            # Constrain to a single observation period if specified
            df = df[np.logical_and(df[0] > mjd_range[0],
                                   df[0] < mjd_range[1])]

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
        
        gssub = gs[i].subgridspec(1, 2, width_ratios=[5, 1])
        ax0 = fig.add_subplot(gssub[0,0])
        ax1 = fig.add_subplot(gssub[0,1])

        # Plot Data
        ax0.errorbar(mjd, flux , yerr=err, ls='none', marker=".", color=band_colors[i], ms=2, elinewidth=0.75)
        # Plot Model
        t, m, errs = models[i]
        period_pick = np.logical_and(t >=mjd_min,t <= mjd_max)
        t = t[period_pick]
        m = m[period_pick]
        errs = errs[period_pick]
        ax0.plot(t,m, color="black", lw=1)
        ax0.fill_between(t, m+errs, m-errs, alpha=0.5, color="black")
        ax0.set_ylabel("Flux ({})".format(fltr),rotation=0,labelpad=30)
        ax0.set_xlim(mjd_min,mjd_max)
        
        # Plot Time delay posterior distributions
        tau_samples = samples_chunks[i][2]
        ax1.hist(tau_samples, color=band_colors[i], bins=50)
        ax1.axvline(x = np.percentile(tau_samples, [16, 50, 84])[1], color="black",lw=0.5)
        ax1.axvline(x = np.percentile(tau_samples, [16, 50, 84])[0] , color="black", ls="--",lw=0.5)
        ax1.axvline(x = np.percentile(tau_samples, [16, 50, 84])[2], color="black",ls="--",lw=0.5)
        ax1.axvline(x = 0, color="black",ls="--")    
        ax1.yaxis.set_tick_params(labelleft=False)
        ax1.set_xlim(tau_min,tau_max)
        if ccf_data[i] is not None:
            ax1.hist(ccf_data[i], bins = 50, color = 'grey')
        
        if i == ilast:
            ax0.set_xlabel("Time")
            ax0.label_outer()
        else:
            ax1.xaxis.set_tick_params(labelbottom=False)
        
        ax1.set_yticks([])

        if i == 0:
            title_ext = roa_model + ' {}'.format(select_period)
            fig.suptitle('{} Lightcurves {}'.format(config.agn_name(), title_ext))

    plt.subplots_adjust(wspace=0)
    
    print('Writing {}'.format(output_file))
    plt.savefig(output_file)
    plt.show()
#    plt.close()

    return


def CalibrationPlot(model,overwrite=True):

    config = model.config()
    fltrs = config.fltrs()
    
    # All the calibrated lightcurves pre-fitting on the same plot
    
    calib_curve_plot = '{}/Calibrated_LCs.pdf'.format(config.output_dir())
    if Utils.check_file(calib_curve_plot) == True and overwrite==False:
        print('Not running CalibrationPlot, file exists: {}'.format(calib_curve_plot))
        return

    data=[]
    plt.style.use(['seaborn'])
    plt.rcParams.update({
        "font.family": "Sans",  
        "font.serif": ["DejaVu"],
        "figure.figsize":[40,15],
        "font.size": 14})
    fig, axs = plt.subplots(len(fltrs),sharex=True)
    fig.suptitle('{} Calibrated light curves'.format(config.agn_name()))
    for i,fltr in enumerate(fltrs):
        calib_file = '{}/{}_{}.dat'.format(config.output_dir(),config.agn_name(),fltr)
        data.append(pd.read_csv(calib_file,
                                header=None,index_col=None,
                                quoting=csv.QUOTE_NONE,delim_whitespace=True).sort_values(0))
        mjd = data[i][0]
        flux = data[i][1]
        err = data[i][2]
        axs[i].errorbar(mjd, flux , yerr=err, ls='none', marker=".", ms=3.5, elinewidth=0.5)
        axs[i].set_ylabel('{} filter flux'.format(fltr))

    axs[-1].set_xlabel('Time (days, MJD)')
    print('Writing {}'.format(calib_curve_plot))
    plt.savefig(calib_curve_plot)
    plt.close()

    return


def CalibrationSNR(model,select_period=None,overwrite=False):
    print('Running PyROA CalibrationSNR')

    # references for convenience
    config = model.config()
    calib_params = config.calibration_params()
    roa_params = config.roa_params()

    delay_ref = roa_params["delay_ref"]
    sig_level = calib_params['sig_level']
    exclude_fltrs = roa_params["exclude_fltrs"]    
    fltrs = config.fltrs()

    fltrs = [fltr for fltr in fltrs if fltr not in exclude_fltrs and fltr != delay_ref]    
    fltrs = [delay_ref] + fltrs
    
    add_ext = '_{}'.format(roa_params['model'])
    
    data=[]

    mjd_range = None
    # We might chose to display SNR for a single obervation period
    if select_period:
        period_to_mjd_range = config.observation_params()['periods']
        if select_period not in period_to_mjd_range:
            raise Exception('Error: selected period {} not in observation periods for {}, check config'.format(select_period,config.agn_name()))
        mjd_range = config.observation_params()['periods'][select_period]['mjd_range']
        add_ext = add_ext + ' {}'.format(select_period)

    snr = []
    for fltr in fltrs:
        calib_file = '{}/{}_{}.dat'.format(config.output_dir(),config.agn_name(),fltr)
    
        if Utils.check_file(calib_file,exit=True):
            # get mjd flux err from the calibration file as a numpy array of first three columns
            df = pd.read_csv(calib_file,
                             header=None,index_col=None,
                             quoting=csv.QUOTE_NONE,
                             delim_whitespace=True).sort_values(0).loc[:,0:2]
            if mjd_range:
                df = df[np.logical_and(df[0] > mjd_range[0],
                                       df[0] < mjd_range[1])]
            
            snr.append(Utils.signal_to_noise(df,sig_level,fltr))

    ext = 'for AGN {} {}'.format(config.agn_name(),add_ext)
    print('Signal to Noise ratio by filter {}'.format(ext))
    print(tabulate([snr],headers=fltrs))


def ConvergencePlot(model,select_period=None,overwrite=False):

    config = model.config()
    roa_params = config.roa_params()

    Nsamples = roa_params["Nsamples"]
    Nburnin = roa_params["Nburnin"]

    mjd_range = None
    
    # tau initialisation by filter
    delay_ref = roa_params["delay_ref"]
    exclude_fltrs = roa_params["exclude_fltrs"]    
    fltrs = config.fltrs()
    
    fltrs = [fltr for fltr in fltrs if fltr not in exclude_fltrs and fltr != delay_ref]
    fltrs = [delay_ref] + fltrs

    add_ext = '_{}'.format(roa_params['model'])
    if select_period:
        add_ext = add_ext = '_{}'.format(select_period)

    samples_file = '{}/samples_flat{}.obj'.format(config.output_dir(),add_ext)
    if Utils.check_file(samples_file) == False:
        input_ext = '_{}'.format(roa_params['model'])
    else:
        input_ext = add_ext
    
    filehandler = open('{}/samples_flat{}.obj'.format(config.output_dir(),input_ext),"rb")
    samples = pickle.load(filehandler)

    output_file = '{}/ROA_Convergence{}.pdf'.format(config.output_dir(),add_ext)
    if Utils.check_file(output_file) == True and overwrite==False:
        print('Not running ROA ConvergencePlot, file exists: {}'.format(output_file))
        return
    
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

    plt.savefig(output_file)
    plt.show()
#    plt.close()
    
    return

def ChainsPlot(model,select='tau',select_period=None,start_sample=0,overwrite=False):
    config = model.config()
    roa_params = config.roa_params()

    delay_ref = roa_params["delay_ref"]
    roa_model = roa_params["model"]
    Nburnin = roa_params["Nburnin"]
    mjd_range = None

    # We might chose to run the ROA for a single obervation period
    if select_period:
        period_to_mjd_range = config.observation_params()['periods']
        if select_period not in period_to_mjd_range:
            raise Exception('Error: selected period {} not in observation periods for {}, check config'.format(select_period,config.agn_name()))
        mjd_range = config.observation_params()['periods'][select_period]['mjd_range']

    exclude_fltrs = roa_params["exclude_fltrs"]    
    fltrs = config.fltrs()

    fltrs = [fltr for fltr in fltrs if fltr not in exclude_fltrs and fltr != delay_ref]
    fltrs = [delay_ref] + fltrs

    if len(fltrs) == 0:
        raise Exception('Insufficient filter bands passed to PyROA FitPlot: {} with reference filter {}'.format(fltrs,delay_ref))
    
    add_ext = '_{}'.format(roa_model)

    if select_period:
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
    plt.show()
#    plt.close()
    return

def CornerPlot(model,select='tau',select_period=None,overwrite=False):
    config = model.config()
    roa_params = config.roa_params()

    delay_ref = roa_params['delay_ref']
    roa_model = roa_params['model']
    Nburnin = roa_params['Nburnin']
    mjd_range = None

    # We might chose to run the ROA for a single obervation period
    if select_period:
        period_to_mjd_range = config.observation_params()['periods']
        if select_period not in period_to_mjd_range:
            raise Exception('Error: selected period {} not in observation periods for {}, check config'.format(select_period,config.agn_name()))
        mjd_range = config.observation_params()['periods'][select_period]['mjd_range']

    exclude_fltrs = roa_params['exclude_fltrs']    
    fltrs = config.fltrs()

    fltrs = [fltr for fltr in fltrs if fltr not in exclude_fltrs and fltr != delay_ref]
    fltrs = [delay_ref] + fltrs

    if len(fltrs) == 0:
        raise Exception('Insufficient filter bands passed to PyROA FitPlot: {} with reference filter {}'.format(fltrs,delay_ref))
    
    add_ext = '_{}'.format(roa_model)
    if select_period:
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
    plt.show()
#    plt.close()
    return
