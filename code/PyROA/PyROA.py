import os
import PyROA.Utils as PUtils
from multiprocessing import Pool
from itertools import chain
from tabulate import tabulate
import numpy as np
import pandas as pd
import emcee
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class LCModel():

    # The priors are for the extra error parameter and delta,
    # which are uniform where the limits must be specified in the following way:
    # priors = [[delta_lower, delta_upper], [sig_lower, sig_upper]]
    def __init__(self, root_dir, agn_name, init_delta=1.0, sig_level = 4.0,
                 Nsamples=15000, Nburnin=10000,priors=[[0.01, 10.0], [0.0, 2.0]]):

        # manage lightcurve (LCO) data, check available scope and filters
        output_dir = '{}/{}/output'.format(root_dir,agn_name)
        PUtils.check_and_create_dir(output_dir)    
        lco_lc_file = PUtils.load_lco_lightcurves(root_dir,agn_name)
        fltrs, scopes = PUtils.write_scope_filter_data(agn_name,lco_lc_file, output_dir)

        self._output_dir = output_dir
        self._agn_name   = agn_name
        self._fltrs      = fltrs
        self._scopes     = scopes
        
        self._priors     = priors
        self._init_delta = init_delta
        self._sig_level  = sig_level
        self._nsamples   = Nsamples
        self._nburnin    = Nburnin


    def output_dir(self): return self._output_dir
    def agn_name(self): return self._agn_name
    def fltrs(self): return self._fltrs
    def scopes(self): return self._scopes
        
    def InterCalibrateFilt(self,fltr):

        scopes_array = []
        data=[]
        for i in range(len(self._scopes)):
            scope_file = '{}/{}_{}_{}.dat'.format(self._output_dir,self._agn_name,fltr,self._scopes[i])
            #Check if file is empty
            if os.stat(scope_file).st_size == 0:
                print("")
            else:
                data.append(np.loadtxt(scope_file))
                scopes_array.append([self._scopes[i]]*np.loadtxt(scope_file).shape[0])
            
        scopes_array = [item for sublist in scopes_array for item in sublist]

        output_file = '{}/{}_{}.dat'.format(self._output_dir,self._agn_name,fltr)

        if os.path.exists(output_file) == True:
            return

        # no calibration data for this filter exists
        ########################################################################################    
        #Run MCMC to fit to data
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
        
        pos_chunks[-1][0] = self._init_delta          #Initial delta
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
        with Pool() as pool:

            sampler = emcee.EnsembleSampler(nwalkers, ndim, PUtils.log_probability_calib, 
                                            args=(data, self._priors, self._sig_level, init_params_chunks), pool=pool)
            sampler.run_mcmc(pos, self._nsamples, progress=True);

        raise Exception('no further')

        #Extract samples with burn-in of 1000
        samples_flat = sampler.get_chain(discard=_self.nburnin, thin=15, flat=True)
                
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
            model = interp(mjd)
                
            #Sigma Clipping
            mask = (abs(model - flux) < self._sig_level*err)
        
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
                if (abs(model[j] - flux[j]) > self._sig_level*err[j]):
                    Calibrated_err.append((abs(model[j] - flux[j])/self._sig_level))
                else:
                    Calibrated_err.append(err[j])
                
        Calibrated_mjd = np.array(Calibrated_mjd)
        Calibrated_flux = np.array(Calibrated_flux)
        Calibrated_err = np.array(Calibrated_err)
                
        print("<A> = ", np.mean(A_values))
        print("<B> = ", np.mean(B_values))
    
        #Model
        interp = interpolate.interp1d(t, m, kind="linear", fill_value="extrapolate")
        model_j1 = interp(Calibrated_mjd)

        print(model_j1.shape)
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
            'f5':model_j1,
            'f6':error_j1
        })
        df.to_csv(output_file,
                  header=False,sep=' ',float_format='%25.15e',index=False,
                  quoting=csv.QUOTE_NONE,escapechar=' ')
        return

    def InterCalibrateFiltPlot(self,fltr,plot_corner=False):

        calib_file = '{}/{}_{}.dat'.format(self._output_dir,self._agn_name,fltr)

        if os.path.exists(calib_file) == True:
            df = pd.read_csv(calib_file,
                             header=None,index_col=None,
                                 quoting=csv.QUOTE_NONE,delim_whitespace=True)

            output_file = '{}/{}_Calibration_Plot.pdf'.format(datadir,self.fltr)
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

            output_file = '{}/{}_Calibration_CornerPlot.pdf'.format(datadir,self.fltr)
            
            if os.path.exists(output_file) == False and  plot_corner == True:
                plt.rcParams.update({'font.size': 15})
                #Save Cornerplot to figure
                fig = corner.corner(samples_flat, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True,
                                    title_kwargs={"fontsize": 20}, truths=params);
                print('Writing calibration corner plot {}'.format(output_file))
                plt.savefig(output_file)
                plt.close();
           
        return
