import os,json
from AGNLCLib import Utils
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
import scipy.interpolate as interpolate
import scipy.special as special
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import quad


def InterCalibrateFilt(model,fltr,overwrite=False):
    print('Running PyROA InterCalibrateFilt for filter {}'.format(fltr))

    # references for convenience
    config = model.config()
    params = config.calibration_params()

    # local variables
    scopes_array = []
    data=[]
        
    for scope in config.scopes():
        scope_file = '{}/{}_{}_{}.dat'.format(config.output_dir(),config.agn_name(),fltr,scope)
        #Check if file is empty
        if os.stat(scope_file).st_size == 0:
            print("")
        else:
            data.append(np.loadtxt(scope_file))
            scopes_array.append([scope]*np.loadtxt(scope_file).shape[0])
            
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
        
    pos_chunks[-1][0] = params['init_delta']
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
    sig_level = params['sig_level']
    
    # 12 threads works if 12 virtual cores available
    # Reduce memory usage -> 6 threads 2023/06/14 as turgon has very little memory(?)
    with Pool(8) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, Utils.log_probability_calib, 
                                        args=(data, [params['delta_prior'], params['sigma_prior']],
                                              sig_level, init_params_chunks), pool=pool)
        sampler.run_mcmc(pos, params['Nsamples'], progress=True);

    #Extract samples with burn-in of 10000 (default setting, see global.json)
    samples_flat = sampler.get_chain(discard=params['Nburnin'], thin=15, flat=True)
                
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

    # read calibration file which should now exist
    calib_file = '{}/{}_{}.dat'.format(config.output_dir(),config.agn_name(),fltr)
    
    if os.path.exists(calib_file) == True:
        df = pd.read_csv(calib_file,
                         header=None,index_col=None,
                         quoting=csv.QUOTE_NONE,delim_whitespace=True).sort_values(0)

        output_file = '{}/{}_Calibration_Plot.pdf'.format(config.output_dir(),fltr)
        if (os.path.exists(output_file) == False) or (overwrite == True):
            
            plt.rcParams.update({
                "font.family": "Sans", 
                "font.serif": ["DejaVu"],
                "figure.figsize":[20,10],
                "font.size": 20})          
        
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
            plt.close()

        output_file = '{}/{}_Calibration_CornerPlot.pdf'.format(config.output_dir(),fltr)
            
        if (os.path.exists(output_file) == False) or (overwrite == True):
            plt.rcParams.update({'font.size': 15})
            #Save Cornerplot to figure
            fig = corner.corner(samples_flat, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True,
                                title_kwargs={"fontsize": 20}, truths=params);
            print('Writing calibration corner plot {}'.format(output_file))
            plt.savefig(output_file)
            plt.close();
           
    return

def Fit(model):
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
    plot_corner = roa_params["plot_corner"]
    accretion_disk = roa_params["accretion_disk"]
    include_slow_comp = roa_params["include_slow_comp"]
    slow_comp_delta = roa_params["slow_comp_delta"]
    delay_dist = roa_params.get("delay_dist",False)
    select_period = roa_params.get("select_period",None)
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

    # We might chose to run the ROA for a single obervation period
    if select_period:
        period_to_mjd_range = config.observation_params()['periods']
        if select_period not in period_to_mjd_range:
            raise Exception('Error: selected period {} not in observation periods for {}, check config'.format(select_period,config.agn_name()))
        mjd_range = config.observation_params()['periods'][select_period]['mjd_range']
        
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
    # Reduce memory usage -> 4 threads 2023/06/18

    with Pool(4) as pool:

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
            if (i>0):
                tau_rms = np.percentile(samples_chunks[i][3], [16, 50, 84])[1]
                params.append([tau_rms])
            else:
                tau_rms=0.0
                params.append([0.0])
           # else:
               # tau_rms =  np.percentile(samples_chunks[-1][1], [16, 50, 84])[1]*(((wavelengths[i]/wavelengths[0]) - 0.999)** np.percentile(samples_chunks[-1][2], [16, 50, 84])[1])               
            
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
    add_ext = '_{}'.format(roa_params['model'])
    if mjd_range:
        add_ext = add_ext + '_{}'.format(select_period)
    filehandler = open('{}/samples_flat{}.obj'.format(config.output_dir(),add_ext),"wb")
    pickle.dump(samples_flat,filehandler)
    filehandler = open('{}/samples{}.obj'.format(config.output_dir(),add_ext),"wb")
    pickle.dump(samples,filehandler)

    filehandler = open('{}/X_t{}.obj'.format(config.output_dir(),add_ext),"wb")
    pickle.dump([t, m, errs],filehandler)
    if (include_slow_comp==True):      
        filehandler = open('{}/Slow_comps{}.obj'.format(config.output_dir(),add_ext),"wb")
        pickle.dump(slow_comps_out,filehandler)
        
    filehandler = open('{}/Lightcurves_models{}.obj'.format(config.output_dir(),add_ext),"wb")
    pickle.dump(models,filehandler)
    
    if plot_corner:
        #Plot Corner Plot
        plt.rcParams.update({'font.size': 15})
        #Save Cornerplot to figure
        fig = corner.corner(samples_flat, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 20});
        fig.savefig('{}/ROACornerPlot_{}{}.pdf'.format(config.output_dir(),params["model"],add_ext),"wb")
        plt.close();
        
    return samples, samples_flat, t, m, errs, slow_comps_out, params, models


########################################
# Diagnostic Graphs                    #
########################################

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
        "font.size": 40})
    fig, axs = plt.subplots(len(fltrs),sharex=True)
    fig.suptitle('{} Calibrated light curves'.format(config.agn_name()))
    for i,fltr in enumerate(fltrs):
        calib_file = '{}/{}_{}.dat'.format(config.output_dir(),config.agn_name(),fltr)
        data.append(pd.read_csv(calib_file,
                                header=None,index_col=None,
                                quoting=csv.QUOTE_NONE,delim_whitespace=True)).sort_values(0)
        mjd = data[i][0]
        flux = data[i][1]
        err = data[i][2]
        axs[i].errorbar(mjd, flux , yerr=err, ls='none', marker=".", ms=3.5, elinewidth=0.5)
        axs[i].set_ylabel('{} filter flux'.format(fltr))

    axs[-1].set_xlabel('Time (days, MJD)')

    print('Writing {}'.format(calib_curve_plot))
    plt.savefig(calib_curve_plot)


