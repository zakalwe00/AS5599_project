import os
import numpy as np
import math
import pandas as pd
import scipy.interpolate as interpolate
from numba import jit
from numba import prange

@jit(nopython=True, cache=True, parallel=True)
def CalculatePorc(t_data, Flux, Flux_err, delta):

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

    return Ps

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
        data_pick = np.where(np.absolute(t[j]-t_data) < 5.0*delta)[0]
        t_data_use = t_data[data_pick]
        Flux_err_use = Flux_err[data_pick]
        Flux_use = Flux[data_pick]
        
        if (len(data_pick)<1):
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
def _log_likelihood_calib(params, data, sig_level):

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
        interp = interpolate.interp1d(t, m_scaled, kind="linear", fill_value="extrapolate")
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
def _log_prior_calib(params, priors, s, init_params_chunks):
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

        # back to original for Fairall testing HC 20230619
#        A_prior.append(-2.0*np.log(lnA_prior_width*A*np.sqrt(2.0*np.pi)) - (np.log(A)/lnA_prior_width)**2.0)
#        B_prior.append(2.0*np.log((1.0/np.sqrt(2.0*np.pi*(B_prior_width**2)))*np.exp(-0.5*(B/B_prior_width)**2)))
        A_prior.append(-2.0*np.log(lnA_prior_width*np.sqrt(2.0*np.pi)) - (np.log(A)/lnA_prior_width)**2.0)
        B_prior.append(-2.0*np.log(B_prior_width*np.sqrt(2.0*np.pi)) - (B/B_prior_width)**2)
        
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
    lp = _log_prior_calib(params, priors, len(data), init_params_chunks)
    if not np.isfinite(lp):
        return -np.inf
    return lp + _log_likelihood_calib(params, data, sig_level)

def _log_prior(params, priors, add_var, data, delay_dist, AccDisc, wavelengths, init_params_chunks):
    Nchunk = 2
    if (AccDisc == False):
        Nchunk+=1
    
    if (add_var == True):
        Nchunk +=1
    if (delay_dist == True and AccDisc == False):
        Nchunk+=1
        
    Npar =  Nchunk*len(data) + 1    
    if (AccDisc==True):
        Npar =  Nchunk*len(data) + 3   
    chunk_size = Nchunk

    #Break params list into chunks of 3 i.e A, B, tau in each chunk
    params_chunks = [params[i:i + chunk_size] for i in range(0, len(params), chunk_size)]
    
    #Extract delta and extra variance parameters as last in params list
    delta = params_chunks[-1][0]
        
    #Read in priors
    A_prior = priors[0]
    B_prior = priors[1]
    if (AccDisc == False):
        tau_prior = priors[2]
        delta_prior = priors[3]
        if (add_var == True): 
            V_prior = priors[4]    

    if (AccDisc == True):
        T1_prior = priors[-3]
        b_prior = priors[-2]
        T1 = params_chunks[-1][0]
        b = params_chunks[-1][1]
        if (add_var == True): 
            V_prior = priors[2]
        delta = params_chunks[-1][2]
        delta_prior=priors[-1]

    check=[]
    #V_priors=np.empty(len(data))
    #Loop over lightcurves
    pr=[]
    
    for i in range(len(data)):
        A = params_chunks[i][0]/init_params_chunks[i][0] # Check A as a fraction of inital value to compare with prior
        B = params_chunks[i][1]/init_params_chunks[i][1]
        if (AccDisc == False):
            tau = params_chunks[i][2]
        
        if (add_var == True):
            V =  params_chunks[i][-1]/(init_params_chunks[i][-1]*5)
            
            
        if (delay_dist == True and i>0):
            if (params_chunks[i][3]>=0.0):
                tau_rms = params_chunks[i][3]
                #pr.append(2.0*np.log((1.0/np.sqrt(2.0*np.pi*(rms_prior_width**2)))*np.exp(-0.5*(tau_rms/rms_prior_width)**2)))
                check.append(0.0)
            else:
                check.append(1.0)
            
        #Force peak delays to be larger than blurring reference
        # if (delay_dist == True):
        #     if (tau >=params_chunks[0][2]):
        #         check.append(0.0)
        #     else:
        #         check.append(1.0)
        # else:
        #     pr.append(0.0)

        if (AccDisc == True):
            if T1_prior[0] <= T1 <= T1_prior[1] and b_prior[0] <= b <= b_prior[1]:
                check.append(0.0)
            else:
                check.append(1.0)
        else:
            if tau_prior[0] <= tau <= tau_prior[1]:
                check.append(0.0)
            else:
                check.append(1.0)
             
        if (add_var == True):
            if V_prior[0]<= V <= V_prior[1]:
                check.append(0.0)
            else:
                check.append(1.0)
        
        if A_prior[0] <= A <= A_prior[1] and B_prior[0] <= B <= B_prior[1]:
            check.append(0.0)
        else:
            check.append(1.0)
            
    if np.sum(np.array(check)) == 0.0 and delta_prior[0]<= delta <= delta_prior[1]:
        return 0.0 #+ np.sum(pr)

    else:
        return -np.inf

# Calculate Bayes Information Criterion
def _BIC(params, data, add_var, size, sig_level,include_slow_comp, slow_comp_delta, P_func,
         slow_comps, P_slow, init_delta, delay_dist, psi_types, AccDisc, wavelengths,
         integral, integral2):

    Nchunk = 2
    if (AccDisc == False):
        Nchunk+=1
    
    if (add_var == True):
        Nchunk +=1
    if (delay_dist == True and AccDisc == False):
        Nchunk+=1

    Npar =  Nchunk*len(data) + 1    
    if (AccDisc==True):
        Npar =  Nchunk*len(data) + 3    
      
    chunk_size = Nchunk#int((Npar - 1)/len(data))

    #Break params list into chunks of 3 i.e A, B, tau in each chunk
    params_chunks = [params[i:i + chunk_size] for i in range(0, len(params), chunk_size )] 
    
    #Extract delta and extra variance parameters as last in params list
    if (delay_dist == True and AccDisc == False):
        delta = params_chunks[-1][0]
        rmss = np.zeros(size)
        taus = np.zeros(size)
    
    if (AccDisc == False):
        delta = params_chunks[-1][0]
        wavelengths = [None]*len(data)
        T1 = None 
        b = None
        integral = None
        
    if (AccDisc == True):
        T1 = params_chunks[-1][0]
        b = params_chunks[-1][1]
        rmss = np.zeros(size)
        taus = np.zeros(size)
        delta = params_chunks[-1][2]

    #Loop through each lightcurve and shift data by parameters
    merged_mjd = np.zeros(size)
    merged_flux = np.zeros(size)
    merged_err = np.zeros(size)

    prev=0
    sizes = np.zeros(int(len(data)+1))
    for i in range(len(data)):
        A = params_chunks[i][0]
        B = params_chunks[i][1] 
        if (AccDisc==False):  
            tau = params_chunks[i][2]
        else:
            l_0 = wavelengths[0]
            l = wavelengths[i] - l_0        
            l_delay_ref = wavelengths[0] - l_0
            tau_0 = (l_0*1e-10*1.3806e-23*T1/(6.63e-34*3e8))**(1.0/b)
            tau = tau_0*((l/l_0)**(1.0/b))*8.0*(np.pi**4)/(15.0*integral(b)) - tau_0*((l_delay_ref/l_0)**(1.0/b))*8.0*(np.pi**4)/(15.0*integral(b)) #Measure mean from delay reference
            tau_rms = np.sqrt((tau_0**2)*((l/l_0)**(2.0/b))*integral2(b)/integral(b))
            

        if (add_var == True):
            V =  params_chunks[i][-1]
            
        if (delay_dist == True and AccDisc == False):
            if (i>0):
                tau_rms = params_chunks[i][3]
            else:
                tau_rms=0.0
                
        mjd = data[i][:,0]
        flux = data[i][:,1]
        err = data[i][:,2]
        
        sizes[i+1] = len(mjd)

        #Add extra variance
        if (add_var == True):
            err = np.sqrt((err**2) + (V**2))
                   
        if (include_slow_comp==True):
            t_slow, m_slow, errs_slow = slow_comps[i]
            flux = (flux - B - np.interp(mjd, slow_comps[i][0], slow_comps[i][1]))/A
        else:
            flux = (flux - B )/A   
            P_slow[i]=0.0 
            
        err = err/A
        #Shift data
        mjd = mjd - tau

        #Add shifted data to merged lightcurve
        for j in range(len(mjd)):
            merged_mjd[int(j+ prev)] = mjd[j]
            merged_flux[int(j+ prev)] = flux[j]
            merged_err[int(j+ prev)] = err[j]
            if (delay_dist == True or AccDisc == True):
                rmss[int(j+ prev)] = tau_rms
                taus[int(j+ prev)] = tau
               # factors[int(j+ prev)] = delta/np.sqrt(delta**2 + (tau_rms)**2)#/delta
     
        prev = int(prev + len(mjd))

    # Calculate ROA to merged LC
    if (delay_dist == False and AccDisc == False):

        t, m, errs = RunningOptimalAverage(merged_mjd, merged_flux, merged_err, delta)
        
        #Normalise lightcurve

        m_mean = np.mean(m)# np.average(m, weights = 1.0/(errs**2))

        m_rms = np.std(m)
        m = (m-m_mean)/m_rms
        errs = errs/m_rms
        
        #Calculate no. of parameters
        if (P_func == None):                                              
            P=CalculateP(merged_mjd, merged_flux, merged_err, delta)
        else:
            P = P_func(delta)
         
    #Calculate no. of paramters for delay_dist==True here, actual ROA calcualted in loop per lightcurve   
    else:
        factors, conv, x, d = CalcWinds(merged_mjd, merged_flux, merged_err, delta, rmss, len(data), sizes,  taus, psi_types, wavelengths, T1, b, integral)
        t,m,errs, P = RunningOptimalAverageConv(merged_mjd, merged_flux, merged_err, d, factors, conv, x) 
        P=CalculateP(merged_mjd, merged_flux, merged_err, delta)

    #Calculate chi-squared for each lightcurve and sum
    lps=[0]*len(data)
    prev=0
    for i in range(len(data)):

        A = params_chunks[i][0]
        B = params_chunks[i][1] 
        if (AccDisc==False):  
            tau = params_chunks[i][2]
        else:
            l_0 = wavelengths[0]
            l = wavelengths[i] - l_0        
            l_delay_ref = wavelengths[0] - l_0
            tau_0 = (l_0*1e-10*1.3806e-23*T1/(6.63e-34*3e8))**(1.0/b)
            tau = tau_0*((l/l_0)**(1.0/b))*8.0*(np.pi**4)/(15.0*integral(b)) - tau_0*((l_delay_ref/l_0)**(1.0/b))*8.0*(np.pi**4)/(15.0*integral(b)) #Measure mean from delay reference
            tau_rms = np.sqrt((tau_0**2)*((l/l_0)**(2.0/b))*integral2(b)/integral(b))
        if (add_var == True):
            V =  params_chunks[i][-1]

        #Originial lightcurves
        mjd = data[i][:,0]
        flux = data[i][:,1]
        err = data[i][:,2]
        
        #Add extra variance
        if (add_var == True):
            err = np.sqrt((err**2) + (V**2)) 
            



        if (delay_dist == False and AccDisc == False):
            t_shifted = t + tau
            #interp = interpolate.interp1d(t_shifted, m, kind="linear", fill_value="extrapolate")
            m_m = m#interp(t)
        

                    
            if (include_slow_comp==True):
                m_s = np.interp(t_shifted, slow_comps[i][0], slow_comps[i][1])  #Not sure - originally t not t_shifted
                m_scaled = A*(m_m) + B + m_s
            else:
                m_scaled = A*(m_m) + B

         
            #Model
            interp = interpolate.interp1d(t_shifted, m_scaled, kind="linear", fill_value="extrapolate")
            model = interp(mjd)
            
        #Calculate ROA at mjd of each lightcurve using different delta
        else:
            if (AccDisc == False):
                if (i>0):
                    tau_rms = params_chunks[i][3]
                else:
                    tau_rms=0.0
            #else:
             #   tau_rms = params_chunks[-1][1]*(((wavelengths[i]/wavelengths[0]) - 0.999)**params_chunks[-1][2])

            
            
            
            m_mean = np.mean(m[prev : int(prev + len(mjd))])
            m_rms = np.std(m[prev : int(prev + len(mjd))])
            m[prev : int(prev + len(mjd))] = (m[prev : int(prev + len(mjd))]-m_mean)/m_rms
            errs[prev : int(prev + len(mjd))] = errs[prev : int(prev + len(mjd))]/m_rms  
             
            Xs = m[prev : int(prev + len(mjd))]
            errs = errs[prev : int(prev + len(mjd))]
            
            
            if (include_slow_comp==True):
                m_s = np.interp(mjd, slow_comps[i][0], slow_comps[i][1])
                
                model = A*Xs + B + m_s
            else:
                model = A*Xs + B
                     
        prev = int(prev + len(mjd))
            

        chi2 = np.empty(len(mjd))
        ex_term = np.empty(len(mjd))  
        for j in range(len(mjd)):

            if(abs(model[j]-flux[j]) < sig_level*err[j]):
            
            
                chi2[j] = ((model[j]-flux[j])**2)/(err[j]**2)
                
                ex_term[j] = np.log(((err[j]**2)/(data[i][j,2]**2)))  
                              
            else:
                chi2[j] =sig_level**2
                ex_term[j] = np.log(((abs(model[j] - flux[j])/sig_level)**2)/(data[i][j,2]**2))
        lps[i]=np.sum(chi2 + ex_term) 
    
    lprob = np.sum(lps)  
    

    #Calculate Penalty
    Penalty = 0.0
    for i in range(len(data)):
        mjd = data[i][:,0]                
        Penalty = Penalty + float(chunk_size+P_slow[i] - 1.0)*np.log(len(mjd))
                    
    if (AccDisc == True):            
        Penalty = Penalty + ((P+2.0)*np.log(len(merged_flux)))
    else:
        Penalty = Penalty + (P*np.log(len(merged_flux)))   
        
    BIC =  lprob + Penalty

    if (math.isnan(BIC) == True):
        return -np.inf
    else:
        return BIC
    
#Probability
def log_probability(params, data, priors, add_var, size, sig_level, include_slow_comp, slow_comp_delta,P_func,
                    slow_comps, P_slow, init_delta, delay_dist, psi_types, AccDisc, wavelengths,
                    integral, integral2, init_params_chunks):
    #Insert t1 as zero for syntax
    Nchunk = 2
    if (AccDisc == False):
        Nchunk+=1
    
    if (add_var == True):
        Nchunk +=1
    if (delay_dist == True and AccDisc == False):
        Nchunk+=1
        params=np.insert(params, [2], [0.0])    #Insert zero for reference delay dist

    Npar =  Nchunk*len(data) + 1    
    if (AccDisc==True):
        Npar =  Nchunk*len(data) + 3  
        
    pos = 2
    if (AccDisc == False):
        params=np.insert(params, pos, [0.0])    #Insert zero for reference delay     

    lp = _log_prior(params, priors, add_var, data, delay_dist,  AccDisc, wavelengths, init_params_chunks)
    if not np.isfinite(lp):
        return -np.inf
    return lp - _BIC(params, data, add_var, size, sig_level, include_slow_comp, slow_comp_delta,P_func, slow_comps, P_slow, init_delta, delay_dist,psi_types, AccDisc, wavelengths, integral, integral2)


########################################
# Maths Operations                     #
########################################

def signal_to_noise(df, sig_level, fltr):
    # remove point with very large error
    filter_large_sigma(df, sig_level, fltr, noprint=True)
    flux = df.loc[:,1]
    err = df.loc[:,2]
    goodvals = df.loc[:,2] != 0.0
    numgood = goodvals.sum()

    m = df.loc[goodvals,1]
    err = df.loc[goodvals,2]
    
    #Normalise lightcurve
    m_mean = np.mean(m)
    m_rms = np.std(m)
    m = (m-m_mean)/m_rms
    err = err/m_rms
    
    snr = np.mean(np.abs(m)/err)
    
    print('Calculated mean SNR={:.3f} for filter {} based on {} observations'.format(snr,fltr,numgood))
    return snr
    

# Automated windowing procedure following Sokal (1989)
def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

# Following the suggestion from Goodman & Weare (2010)
def autocorr_gw2010(y, c=5.0):
    f = autocorr_func_1d(np.mean(y, axis=0))
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]


def autocorr_new(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]

def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i


def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf

def filter_large_sigma(df, sig_level, fltr, noprint=False):
    median_err = np.median(df.loc[:,2])
    goodvals = df.loc[:,2] < median_err * sig_level
    badvals = np.logical_not(goodvals)
    if np.sum(badvals) and noprint is False:
        print('Filtered band {} {} (sig_level={})'.format(fltr,df.loc[badvals,1].to_numpy(),sig_level))
    return df[goodvals]
    
def filter_large_sigma_jumps(df, jump_sig_level, fltr):
    delta1 = np.diff(df.loc[:,2])
    delta1_shift = delta1[1:]
    delta1 = delta1[:-1]
    delta1_limit = np.std(delta1)*jump_sig_level
    # both sides of the datapoint have large sigma flux jumps
    badvals = np.logical_and(np.abs(delta1) > delta1_limit,
                             np.abs(delta1_shift) > delta1_limit)
    # the jumps are in different directions
    badvals = np.logical_and(delta1*delta1_shift < 0.0,badvals)
    badvals = np.concatenate(([False],badvals,[False]))
    goodvals = np.logical_not(badvals)
    if np.sum(badvals):
        print('Filtered band {} {} (jump_sig_level={})'.format(fltr,df.loc[badvals,1].to_numpy(),jump_sig_level))
    return df[goodvals]

# Take and array of observation times and estimate the median cadance,
# rejecting any observations less than 2.5 hours apart (default setting) as these are
# likely to be part of the same observing run
def median_cadence(mjds, min_sep=2.5):
    min_sep_day_frac = 2.5/24.0
    diffs = mjds[1:] - mjds[:-1]
    diffs = diffs[diffs > min_sep_day_frac]
    return np.median(diffs)

########################################
# File Operations                      #
########################################

# Check that the passed file location contains a file
def check_file(cfile,exit=False):
    exists = os.path.exists(cfile)
    if exists and os.path.isfile(cfile) == False:
        raise Exception('Location {} is not a file (as expected)'.format(cfile))
    if exists == False and exit == True:
        raise Exception('Error: expected file at location {}'.format(cfile))
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
    obs = pd.read_csv(obs_file).sort_values('MJD')
    scopes = np.unique(obs.Tel)
    #print('Found telescope list {}'.format(','.join(scopes)))
    fltrs = np.unique(obs.Filter)
    #print('Found filter list {}'.format(','.join(fltrs)))

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
                obs_scope_fltr.to_csv(output_fn, sep=' ', index=False, header=False)

    config.set_fltrs(fltrs)
    config.set_scopes(scopes)

    return

