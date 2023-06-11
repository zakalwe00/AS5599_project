import os
import numpy as np
import pandas as pd
import scipy
from numba import jit
from numba import prange


#Probability
@jit(nopython=True, cache=True, parallel=True)
def RunningOptimalAverage(t_data, Flux, Flux_err, delta):
    #Inputs
    # Flux : Array of data values
    # Flux_err : Array containig errors of data values
    # delta : parameter defining how "loose" memory function is
    # t_data : Array of wavelength data values
    #Outputs
    # t : List of model times 
    # model : List of model fluxes calculated from running optimal average

    gridsize=1000
    
    mx=max(t_data)
    mn=min(t_data)
    length = abs(mx-mn)
    t = np.arange(mn, mx, length/(gridsize))
    model = np.empty(gridsize)
    errs = np.empty(gridsize)
     
    for j in prange(len(t)):

        #Only include significant data points
        t_data_use = t_data[np.where(np.absolute(t[j]-t_data) < 5.0*delta)[0]]
        Flux_err_use = Flux_err[np.where(np.absolute(t[j]-t_data) < 5.0*delta)[0]]
        Flux_use = Flux[np.where(np.absolute(t[j]-t_data) < 5.0*delta)[0]]
        
        if (len(np.where(np.absolute(t[j]-t_data) < 5.0*delta)[0])<1):
            #Define Gaussian Memory Function
            w =  np.exp(-0.5*(((t[j]-t_data)/delta)**2))/(Flux_err**2)
        
            #1/cosh Memory Function
            #w = 1.0/((Flux_err**2)*np.cosh((t[j]-t_data)/delta))
            
            #Lorentzian
            #w = 1.0/((Flux_err**2)*(1.0+((t[j]-t_data)/delta)**2))
          
            #Boxcar
            #w=np.full(len(Flux_err), 0.01) # zero
                      
            w_sum = np.nansum(w)
            #To avoid NAN, 
            if (w_sum==0):
                model[j] = model[j-1]
                errs[j] = errs[j-1]
            else:
                #Calculate optimal average
                model[j] = np.nansum(Flux*w)/w_sum
                #Calculate error
                errs[j] = np.sqrt(1.0/w_sum) 
        else:
            #Define Gaussian Memory Function
            w =np.exp(-0.5*(((t[j]-t_data_use)/delta)**2))/(Flux_err_use**2)
        
            #1/cosh Memory Function
            #w = 1.0/((Flux_err_use**2)*np.cosh((t[j]-t_data_use)/delta))
            
            #Lorentzian
           # w = 1.0/((Flux_err_use**2)*(1.0+((t[j]-t_data_use)/delta)**2))
            
            #Boxcar
            #w = 1.0/(Flux_err_use**2)
            w_sum = np.nansum(w)
            #Calculate optimal average
            model[j] = np.nansum(Flux_use*w)/w_sum
            #Calculate error
            errs[j] = np.sqrt(1.0/w_sum)
        
    return t[0:int(gridsize)], model, errs

@jit(nopython=True, cache=True, parallel=True)
def CalculateP(t_data, Flux, Flux_err, delta):
    Ps = np.empty(len(t_data))
    for i in prange(len(t_data)):
    
        #Only include significant data points
        t_data_use = t_data[np.where(np.absolute(t_data[i]-t_data) < 5.0*delta)[0]]
        Flux_err_use = Flux_err[np.where(np.absolute(t_data[i]-t_data) < 5.0*delta)[0]]
        
        if (len(np.where(np.absolute(t_data[i]-t_data) < 5.0*delta)[0])==0):
            #Define Gaussian Memory Function
            w =np.exp(-0.5*(((t_data[i]-t_data)/delta)**2))/(Flux_err**2)

            #1/cosh Memory function
            #w = 1.0/((Flux_err**2)*np.cosh((t_data[i]-t_data)/delta))

            #Lorentzian
           # w = 1.0/((Flux_err**2)*(1.0+((t_data[i]-t_data)/delta)**2))
           
           #Boxcar
            #w=np.full(len(Flux_err), 0.01)
            
        else:
        
            #Define Gaussian Memory Function
            w =np.exp(-0.5*(((t_data[i]-t_data_use)/delta)**2))/(Flux_err_use**2)

            #1/cosh Memory function
            #w = 1.0/((Flux_err_use**2)*np.cosh((t_data[i]-t_data_use)/delta))

            #Lorentzian
            #w = 1.0/((Flux_err_use**2)*(1.0+((t_data[i]-t_data_use)/delta)**2))
            
            #Boxcar
            #w=1.0/(Flux_err_use**2)
        w_sum = np.nansum(w)

        #P= P + 1.0/((Flux_err[i]**2)*np.nansum(w))
        if (w_sum==0):
            w_sum = 1e-300
        Ps[i] = 1.0/((Flux_err[i]**2)*w_sum)

    return np.nansum(Ps)

#Log Likelihood
def log_likelihood_calib(params, data, sig_level):

    #Break params list into chunks of 3 i.e A, B, V in each chunk
    params_chunks = [params[i:i + 3] for i in range(0, len(params), 3)] 
    
    #Extract delta parameter as last in params list
    delta = params_chunks[-1][0]
    
    #Loop through each lightcurve and shift data by parameters
    merged_mjd = []
    merged_flux = []
    merged_err = []
    avgs=[]
    for i in range(len(data)):
        A = params_chunks[i][0]
        B = params_chunks[i][1]
        sig = params_chunks[i][2] 
        
        mjd = data[i][:,0]
        flux = data[i][:,1]
        err = data[i][:,2]

        #Shift data
        flux = (flux - B)/A
        #Add extra variance
        err = np.sqrt((err**2) + (sig**2))
        
        err = err/A        
       
        #Add shifted data to merged lightcurve
        for j in range(len(mjd)):
            merged_mjd.append(mjd[j])
            merged_flux.append(flux[j])
            merged_err.append(err[j])        
    
    merged_mjd = np.array(merged_mjd)
    merged_flux = np.array(merged_flux)
    merged_err = np.array(merged_err)

    #Calculate ROA to merged lc
    t, m, errs = RunningOptimalAverage(merged_mjd, merged_flux, merged_err, delta)
    P=CalculateP(merged_mjd, merged_flux, merged_err, delta)

    #Calculate chi-squared for each lightcurve and sum
    lps=[0]*len(data)
    for i in range(len(data)):

        A = params_chunks[i][0]
        B = params_chunks[i][1] 
        sig = params_chunks[i][2] 

        #Originial lightcurves
        mjd = data[i][:,0]
        flux = data[i][:,1]
        err = data[i][:,2]
        #Add extra variance
        err = np.sqrt((err**2) + (sig**2))
        
        #Scale and shift model        
        m_scaled = A*(m) + B        
         
        #Model
        interp = scipy.interpolate.interp1d(t, m_scaled, kind="linear", fill_value="extrapolate")
        model = interp(mjd)
        chi2 = np.empty(len(mjd))
        ex_term = np.empty(len(mjd))  
        for j in range(len(mjd)):
            if(abs(model[j]-flux[j]) < sig_level*err[j]):
                chi2[j] = ((model[j]-flux[j])**2)/(err[j]**2)
                ex_term[j] = np.log(2.0*np.pi*(err[j]**2))
            else:
                chi2[j] =sig_level**2
                ex_term[j] = np.log(2.0*np.pi*((abs(model[j] - flux[j])/sig_level)**2))
        lps[i]=np.sum(chi2 + ex_term) 
    
    lprob = np.sum(lps)  
    
    #Calculate Penalty
    Penalty = 0.0
    for i in range(len(data)):
        mjd = data[i][:,0]

        Penalty = Penalty + 3.0*np.log(len(mjd))
            
    Penalty = Penalty + (P*np.log(len(merged_flux)))
        
    BIC =  lprob + Penalty

    return -1.0*BIC

#Calibration priors
def log_prior_calib(params, priors, s, init_params_chunks):
    #Break params list into chunks of 3 i.e A, B, sigma in each chunk
    params_chunks = [params[i:i + 3] for i in range(0, len(params), 3)]
    
    #Extract delta and extra variance parameters as last in params list
    delta = params_chunks[-1][0]

    #Read in priors
    sig_prior = priors[1]
    delta_prior = priors[0]
    A=[]
    B=[]
    V0=[]
    
    check=[]
    A_prior = []
    B_prior=[]
    #Loop over lightcurves
    for i in range(s):
        A = params_chunks[i][0]
        B = params_chunks[i][1]
        sig = params_chunks[i][2]/(init_params_chunks[i][2]*5.0)
        
        B_prior_width=0.5 # mJy
        lnA_prior_width=0.02 # 0.02 = 2%
        
        A_prior.append(-2.0*np.log(lnA_prior_width*A*np.sqrt(2.0*np.pi)) - (np.log(A)/lnA_prior_width)**2.0)
        B_prior.append(2.0*np.log((1.0/np.sqrt(2.0*np.pi*(B_prior_width**2)))*np.exp(-0.5*(B/B_prior_width)**2)))
        
        if sig_prior[0] < sig < sig_prior[1]:
            check.append(0.0)
        else:
            check.append(1.0)
            
    A_prior = np.array(A_prior)
    B_prior = np.array(B_prior)    
          
    if np.sum(np.array(check)) == 0.0 and delta_prior[0]< delta < delta_prior[1]:
        return np.sum(A_prior) + np.sum(B_prior)
    else:
        return -np.inf

def log_probability_calib(params, data, priors, sig_level, init_params_chunks):
    lp = log_prior_calib(params, priors, len(data), init_params_chunks)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_calib(params, data, sig_level)

# Check that the passed file location contains a file
def check_file(cfile):
    exists = os.path.exists(cfile)
    if exists and os.path.isfile(cfile) == False:
        raise Exception('Location {} is not a file (as expected)'.format(cfile))
    return exists

# Check that the passed file location contains a directory
def check_dir(cdir):
    exists = os.path.exists(cdir)
    if exists and os.path.isfile(cdir):
        raise Exception('Location {} is not a directory (as expected)'.format(cdir))
    return exists
                        
def check_and_create_dir(cdir):
    if check_dir(cdir) == False:
        print('Creating directory {}'.format(cdir))
        os.makedirs(cdir)
    return

def write_scope_filter_data(config,obs_file):
    # split LCO file data for this AGN into records by telescope/filter (spectral band)
    obs = pd.read_csv(obs_file)
    scopes = np.unique(obs.Tel)
    print('Found telescope list {}'.format(','.join(scopes)))
    fltrs = np.unique(obs.Filter)
    print('Found filter list {}'.format(','.join(fltrs)))

    # prepare data file per telescope/filter if not already done
    for scope in scopes:
        for fltr in fltrs:
            output_fn = '{}/{}_{}_{}.dat'.format(config.output_dir(), config.agn_name(), fltr, scope)
            if os.path.exists(output_fn) == False:
                # Select time/flux/error per scope and filter
                obs_scope = obs[obs['Tel'] == scope]
                try:
                    obs_scope_fltr  = obs_scope[obs_scope['Filter'] == fltr].loc[:,['MJD','Flux','Error']]
                except KeyError:
                    print('Filter {} not found for telescope {}'.format(fltr, scope))
                    continue
                print('Writing file {}'.format(output_fn))
                obs_scope_fltr.to_csv(output_fn, sep=' ', index=False, header=False)

    config.set_fltrs(fltrs)
    config.set_scopes(scopes)

    return

