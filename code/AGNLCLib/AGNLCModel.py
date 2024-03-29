import os,json,datetime,socket
#import AGNLCLib.Utils as Utils
from . import Utils


# AGN lightcurve model config holds configuration data for the pipeline
# but does not manipulate data or create directories
class AGNLCModelConfig():
        
    def __init__(self, PROJECTDIR, CONFIGDIR, agn_name, output_ext=None):

        if Utils.check_dir(PROJECTDIR) == False:
            raise Exception('Unable to configure AGN lightcurve model, PROJECTDIR={} does not exist'.format(PROJECTDIR))
        global_config_file = '{}/global.json'.format(CONFIGDIR)
        if Utils.check_file(global_config_file) == False:
            raise Exception('Unable to configure AGN lightcurve model, {} does not exist'.format(global_config_file))
        is_turgon = socket.gethostname() == 'turgon'

        # Take parameters from global config file and override with AGN object-specific settings
        with open(global_config_file, 'r') as fd:
            params = json.load(fd)
        self._data_params = params.get('data',{})
        self._observation_params = params.get('observation',{})
        self._calibration_params = params.get('calibration',{})
        self._ccf_params = params.get('CCF',{})
        self._roa_params = params.get('ROA',{})

        #override with any object-specific parameters
        object_config_file = '{}/{}.json'.format(CONFIGDIR,agn_name)
        if Utils.check_file(object_config_file):
            with open(object_config_file, 'r') as fd:
                object_params = json.load(fd)
                if 'data' in object_params:
                    self._data_params.update(object_params['data'])
                if 'observation' in object_params:
                    self._observation_params.update(object_params['observation'])
                if 'calibration' in object_params:
                    self._calibration_params.update(object_params['calibration'])
                if 'CCF' in object_params:
                    self._ccf_params.update(object_params['CCF'])
                if 'ROA' in object_params:
                    self._roa_params.update(object_params['ROA'])

        self._agn_name = agn_name
        self._agn = self._observation_params.get('AGN',agn_name)
        roa_model_name = self._roa_params['model']
        self._roa_params.update(params.get('ROA_{}'.format(roa_model_name)))
        if 'ROA_{}'.format(roa_model_name) in object_params:
            self._roa_params.update(object_params['ROA_{}'.format(roa_model_name)])

        # don't allow turgon to run too many cpus (AWS can to 16, turgon typically 8)
        if is_turgon and self._calibration_params['Nparallel'] > 10:
            self._calibration_params['Nparallel'] = int(float(self._calibration_params['Nparallel'])/2.0)
        if is_turgon and self._roa_params['Nparallel'] > 10:
            self._roa_params['Nparallel'] = int(float(self._roa_params['Nparallel'])/2.0)        

        self._root_dir = '{}/{}'.format(PROJECTDIR,agn_name)
        self._output_dir = '{}/{}/output'.format(PROJECTDIR,agn_name)
        if output_ext is not None:
            self._output_dir = '{}.{}'.format(self._output_dir,output_ext)

        # may want to override these arrays in the config
        self._fltrs       = []
        self._calib_fltrs = []
        self._scopes      = []    

        
    def agn_name(self): return self._agn_name
    def agn(self): return self._agn
    def root_dir(self): return self._root_dir
    def output_dir(self): return self._output_dir
    def fltrs(self): return self._calib_fltrs
    def scopes(self): return self._scopes
    def data_params(self): return self._data_params
    def calibration_params(self): return self._calibration_params
    def observation_params(self): return self._observation_params
    def ccf_params(self): return self._ccf_params
    def roa_params(self): return self._roa_params
    def tmp_dir(self):
        tmp_dir = '{}/output/tmp/{}'.format(self._root_dir,datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        Utils.check_and_create_dir(tmp_dir)
        self._dump_params(tmp_dir)
        return tmp_dir
    def set_scopes(self, scopes): self._scopes = scopes
    def set_fltrs(self, fltrs):
        self._fltrs = fltrs
        self.set_calib_fltrs(fltrs)
    def set_calib_fltrs(self,fltrs):
        calib_fltr_order =  self._calibration_params.get("calib_fltr_order",[])
        calib_fltr_exclude =  self._calibration_params.get("exclude_fltrs",[])
        self._calib_fltrs = [ff for ff in calib_fltr_order if (ff in fltrs and ff not in calib_fltr_exclude)]
        self._calib_fltrs = self._calib_fltrs + [ff for ff in fltrs if (ff not in calib_fltr_order and ff not in calib_fltr_exclude)]
    def set_output_dir(self, output_dir): self._output_dir = output_dir
    def _dump_params(self,tmp_dir,noprint=True):
        params = {'data': self._data_params,
                  'observation': self._observation_params,
                  'calibration': self._calibration_params,
                  'CCF': self._ccf_params,
                  'ROA': self._roa_params}
        output_file = '{}/used_params.json'.format(tmp_dir)
        if Utils.check_file(output_file) == True:
            raise Exception('Unable to dump parameters for this run, file exists {}'.format(output_file))
        with open(output_file,"w") as fd:
            json_string = json.dumps(params,indent=4)
            if noprint == False:
                print(json_string)
            fd.write(json_string)        
    
# Model holds configuration parameters and runs lightcurve analysis functions
class AGNLCModel():
    
    def __init__(self, PROJECTDIR, CONFIGDIR, agn_name, noprint=True, output_ext=None):
        self._config = AGNLCModelConfig(PROJECTDIR, CONFIGDIR, agn_name, output_ext=output_ext)
        
        Utils.check_and_create_dir(self.config().output_dir())
        
        # Load local copy of Las Cumbres Observatory data sourced from AVA https://www.alymantara.com/ava/
        # (if available for this AGN)
        lco_lc_file = '{}/{}/LCO/AVA_{}_lco.csv'.format(PROJECTDIR,self.config().agn_name(),self.config().agn())
        if Utils.check_file(lco_lc_file) == False:
            raise Exception('LCO lightcurve file {} does not exist for this AGN. Source this from https://www.alymantara.com/ava/'.format(lco_lc_file))
        #print('Found LCO lightcurve file {}'.format(lco_lc_file))

        # Load data (perform basic sanity check, record available filters and scopes)
        Utils.write_scope_filter_data(self.config(),lco_lc_file,noprint=noprint)


    def remove_fltr_outliers(self,fltr,remove_outliers,noprint=False):
        lco_lc_file = '{}/LCO/AVA_{}_lco.csv'.format(self.config().root_dir(),self.config().agn())
        if Utils.check_file(lco_lc_file) == False:
            raise Exception('LCO lightcurve file {} does not exist for this AGN. Source this from https://www.alymantara.com/ava/'.format(lco_lc_file))
        #print('Found LCO lightcurve file {}'.format(lco_lc_file))

        # Write data to a new set of filter scope files with a tmp file extension
        Utils.write_scope_filter_data(self.config(),lco_lc_file,noprint=noprint,fltr=fltr,
                                      remove_outliers=remove_outliers,
                                      ext='_{}'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
    
    def config(self): return self._config




