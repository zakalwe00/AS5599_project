import os.json
from PyROA import Utils

# AGN lightcurve model config holds configuration data for the pipeline
# but does not manipulate data or create directories
class AGNLCModelConfig():
    
    self._output_dir = ''
    self._log_dir    = ''
    self._agn_name   = ''
    self._fltrs      = []
    self._scopes     = []    
    
    def __init__(self, agn_name, root_dir):

        if check_dir(root_dir) == False:
            raise Exception('Unable to configure AGN lightcurve model, PROJECTDIR={} does not exist'.format(root_dir))

        global_config_file = '{}/config/global.json'.format(root_dir))
        if check_file(global_config_file) == False:
            raise Exception('Unable to configure AGN lightcurve model, {} does not exist'.format(global_config_file))
        # Take parameters from global config file and override with AGN-specific settings
        params = json.loads(global_confifg)
        object_config_file = '{}/config/{}.json'.format(root_dir,agn_name))
        if check_file(object_config_file):
            params.update(json.loads(object_config_file))
                
        self._agn_name = agn_name
        self._output_dir = '{}/{}/output'.format(root_dir,agn_name)
        self._log_dir = '{}/{}/logs'.format(root_dir,agn_name)

        self._data_params = params['data']
        self._calibration_params = params['calibration']

    def agn_name(self): return self._agn_name
    def output_dir(self): return self._output_dir
    def log_dir(self): return self._log_dir
    def fltrs(self): return self._fltrs
    def scopes(self): return self._scopes
    def data_params(self): return self._data_params
    def calibration_params(self): return self._calibration_params

    def setScopes(self, scopes): self._scopes = scopes
    def setFltrs(self, fltrs): self._fltrs = fltrs

class AGNLCModel():

    self._config = None
    
    def __init__(self, root_dir, agn_name, config=AGNLCModelConfig(agn_name, root)):
        self._config = config
        
        PUtils.check_and_create_dir(self.config().output_dir())
        lco_lc_file = PUtils.load_lco_lightcurves(self.config())
        fltrs, scopes = PUtils.write_scope_filter_data(self.config(),lco_lc_file)

        self._config.setFltrs(fltrs)
        self._config.setScopes(scopes)


    def config(self): return self._config
