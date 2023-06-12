import os,json,datetime
import PyROA.Utils as Utils
import numpy as np
from multiprocessing import Pool
from itertools import chain
from tabulate import tabulate
import pandas as pd
import emcee
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# AGN lightcurve model config holds configuration data for the pipeline
# but does not manipulate data or create directories
class AGNLCModelConfig():
        
    def __init__(self, PROJECTDIR, CONFIGDIR, agn_name):

        if Utils.check_dir(PROJECTDIR) == False:
            raise Exception('Unable to configure AGN lightcurve model, PROJECTDIR={} does not exist'.format(PROJECTDIR))
        global_config_file = '{}/global.json'.format(CONFIGDIR)
        if Utils.check_file(global_config_file) == False:
            raise Exception('Unable to configure AGN lightcurve model, {} does not exist'.format(global_config_file))

        # Take parameters from global config file and override with AGN object-specific settings
        with open(global_config_file, "r") as fd:
            params = json.load(fd)
        self._data_params = params.get('data',{})
        self._observation_params = params.get('observation',{})
        self._calibration_params = params.get('calibration',{})
        self._ccf_params = params.get('ccf',{})

        #override with any object-specific parameters
        object_config_file = '{}/{}.json'.format(CONFIGDIR,agn_name)
        if Utils.check_file(object_config_file):
            with open(object_config_file, "r") as fd:
                object_params = json.load(fd)
                if 'data' in object_params:
                    self._data_params.update(object_params['data'])
                if 'observation' in object_params:
                    self._observation_params.update(object_params['observation'])
                if 'calibration' in object_params:
                    self._calibration_params.update(object_params['calibration'])
                if 'ccf' in object_params:
                    self._ccf_params.update(object_params['ccf'])

        self._agn_name = agn_name
        self._root_dir = '{}/{}'.format(PROJECTDIR,agn_name)
        self._output_dir = '{}/{}/output'.format(PROJECTDIR,agn_name)
        
        self._tmp_dir = '{}/{}/output/tmp/{}'.format(PROJECTDIR,agn_name,
                                                     datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))


        # may want to override these arrays in the config
        self._fltrs      = []
        self._scopes     = []    

        
    def agn_name(self): return self._agn_name
    def root_dir(self): return self._root_dir
    def output_dir(self): return self._output_dir
    def tmp_dir(self): return self._tmp_dir
    def fltrs(self): return self._fltrs
    def scopes(self): return self._scopes
    def data_params(self): return self._data_params
    def calibration_params(self): return self._calibration_params
    def observation_params(self): return self._observation_params
    def ccf_params(self): return self._ccf_params

    def set_scopes(self, scopes): self._scopes = scopes
    def set_fltrs(self, fltrs): self._fltrs = fltrs

# Model holds configuration parameters and runs lightcurve analysis functions
class AGNLCModel():
    
    def __init__(self, PROJECTDIR, CONFIGDIR, agn_name):
        self._config = AGNLCModelConfig(PROJECTDIR, CONFIGDIR, agn_name)
        
        Utils.check_and_create_dir(self.config().output_dir())
        Utils.check_and_create_dir(self.config().tmp_dir())
        
        # Load local copy of Las Cumbres Observatory data sourced from AVA https://www.alymantara.com/ava/
        # (if available for this AGN)
        lco_lc_file = '{0}/{1}/LCO/AVA_{1}_lco.csv'.format(PROJECTDIR,agn_name)
        if Utils.check_file(lco_lc_file) == False:
            raise Exception('LCO lightcurve file {} does not exist for this AGN. Source this from https://www.alymantara.com/ava/'.format(lco_lc_file))
        print('Found LCO lightcurve file {}'.format(lco_lc_file))

        # Load data (perform basic sanity check, record available filters and scopes)
        Utils.write_scope_filter_data(self.config(),lco_lc_file)

    def config(self): return self._config




