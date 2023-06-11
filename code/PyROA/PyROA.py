import os,json
import PyROA.Utils as Utils
from multiprocessing import Pool
from itertools import chain
from tabulate import tabulate
import numpy as np
import pandas as pd
import emcee
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def InterCalibrateFilt(model,fltr):
    print('Running PyROA InterCalibrate for filter {}'.format(fltr))

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

    if Utils.check_file(output_file) == True:
        print('Not running filter {} calibration, file exists: {}'.format(fltr, output_file))
        return

    ########################################################################################    
    # No calibration data for this filter exists
    ########################################################################################
    # Before running make a copy of the configuration you are using
    tmp_params_file = '{}/calibration.json'.format(config.tmp_dir())
    print('Backup calibration params to {}'.format(tmp_params_file))
    with open(tmp_params_file, 'w', encoding='utf-8') as fd:
        json.dump(params, fd, ensure_ascii=False, indent=4)
        
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

    assert(False)
    # not using this yet
    with Pool(5) as pool:

        sampler = emcee.EnsembleSampler(nwalkers, ndim, Utils.log_probability_calib, 
                                        args=(data, [params['delta_prior'], params['sigma_prior']],
                                              params['sig_level'], init_params_chunks), pool=pool)
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
    t, m, errs = RunningOptimalAverage(merged_mjd, merged_flux, merged_err, delta)
    
    Calibrated_mjd = []
    Calibrated_flux = []
    Calibrated_err = [] 
                
    Porc=CalculatePorc(merged_mjd, merged_flux, merged_err, delta)
    
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

    print(" >>>>> DELTA <<<<< ",Delta)
        
    # Put all arrays in a pandas dataframe and export
    df = pd.DataFrame({
        'f1':Calibrated_mjd,
        'f2':Calibrated_flux,
        'f3':Calibrated_err,
        'str1':scopes_array,
        'f4':Porc,
        'f5':interpmodel_j1,
        'f6':error_j1
    })
    df.to_csv(output_file,
              header=False,sep=' ',float_format='%25.15e',index=False,
              quoting=csv.QUOTE_NONE,escapechar=' ')
    return

def InterCalibrateFiltPlot(model,fltr):

    calib_file = '{}/{}_{}.dat'.format(model.output_dir(),model.agn_name(),fltr)
    
    if os.path.exists(calib_file) == True:
        df = pd.read_csv(calib_file,
                         header=None,index_col=None,
                         quoting=csv.QUOTE_NONE,delim_whitespace=True)

        output_file = '{}/{}_Calibration_Plot.pdf'.format(model.output_dir(),fltr)
        if os.path.exists(output_file) == False:
            
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
                plt.errorbar(mjd, flux, yerr=err, ls='none', marker=".", label=str(scopes[i]), alpha=0.5)
                
            plt.errorbar(df[0], df[1], yerr=df[2], ls='none', marker=".", color="black", label="Calibrated")

            plt.xlabel("mjd")
            plt.ylabel("Flux")
            plt.legend()
            print('Writing calibration plot {}'.format(output_file))
            plt.savefig(output_file)
            plt.close()

        output_file = '{}/{}_Calibration_CornerPlot.pdf'.format(model.output_dir(),fltr)
            
        if os.path.exists(output_file) == False and  plot_corner == True:
            plt.rcParams.update({'font.size': 15})
            #Save Cornerplot to figure
            fig = corner.corner(samples_flat, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True,
                                title_kwargs={"fontsize": 20}, truths=params);
            print('Writing calibration corner plot {}'.format(output_file))
            plt.savefig(output_file)
            plt.close();
           
    return

def Fit(model, exclude_filters, init_tau = None, init_delta=1.0,
        delay_dist=False , psi_types = None, add_var=True, sig_level = 4.0, 
        Nsamples=10000, Nburnin=5000, include_slow_comp=False, slow_comp_delta=30.0, 
        delay_ref = None, calc_P=False, AccDisc=False, wavelengths=None, 
        use_backend = False, resume_progress = False, plot_corner=False):
    # references for convenience- rework
    config = model.config()
    params = config.calibration_params()
    data=[]
    for fltr in model._fltrs:
        calib_file = '{}/{}_{}.dat'.format(model._output_dir,model._agn_name,fltr)
        if Utils.check_file(calib_file):
            data.append(np.loadtxt(calib_file))

    self.priors= priors
    self.init_tau = init_tau
    self.init_delta=init_delta
    self.add_var = add_var            
            
    self.delay_dist = delay_dist
    if (delay_dist==True):
        self.psi_types = psi_types
        if (psi_types==None):
            self.psi_types = ["Gaussian"]*len(filters)
        else:
            self.psi_types = np.insert(psi_types, [0], psi_types[0])
    else:
        self.psi_types = [None]*len(filters)
        
    self.sig_level = sig_level
    self.Nsamples = Nsamples
    self.Nburnin = Nburnin
        
    if (delay_ref == None):
        self.delay_ref = filters[0]
    else:
        self.delay_ref = delay_ref
    self.delay_ref_pos = np.where(np.array(filters) == self.delay_ref)[0]
    if (init_tau == None):
        self.init_tau = [0]*len(data)
        if (delay_dist == True):
            self.init_tau = [1.0]*len(data)
        if (AccDisc == True):
            self.init_tau = 5.0*(((np.array(wavelengths)/wavelengths[0]))**1.4)
    else:
        Nchunk = 3
        if (self.add_var == True):
            Nchunk +=1
        if (self.delay_dist == True):
            Nchunk+=1
        
        self.init_tau = np.insert(init_tau, self.delay_ref_pos, 0.0)
            
    self.include_slow_comp=include_slow_comp
    self.slow_comp_delta=slow_comp_delta
        
    self.calc_P=calc_P
    self.AccDisc = AccDisc
    self.wavelengths = wavelengths
    self.use_backend = use_backend
    self.resume_progress = resume_progress
    run = FullFit(data, self.priors, self.init_tau, self.init_delta, self.add_var, 
                  self.sig_level, self.Nsamples, self.Nburnin, self.include_slow_comp, 
                  self.slow_comp_delta, self.calc_P, self.delay_dist, self.psi_types, 
                  self.delay_ref_pos, self.AccDisc, self.wavelengths, self.filters, 
                  self.use_backend, self.resume_progress,plot_corner)

    self.samples = run[0]
    self.samples_flat = run[1]
    self.t = run[2]
    self.X = run[3]
    self.X_errs = run[4]
    if (self.include_slow_comp==True):
        self.slow_comps=run[5]
    self.params=run[6]
    self.models = run[7]
