__modules__ = ['Utils','AGNLCModel','PyROA','PyCCF','PyROA_Plot']

from .Utils import RunningOptimalAverage,CalculateP,CalculatePorc,log_probability_calib,log_probability,median_cadence,check_file,check_dir,check_and_create_dir,write_scope_filter_data,filter_large_sigma,autocorr_gw2010,autocorr_new,signal_to_noise

from .AGNLCModel import AGNLCModelConfig,AGNLCModel

from .PyROA import InterCalibrateFilt,Fit

from .PyROA_Plot import InterCalibratePlot,CalibrationSNR,CalibrationOutlierPlot,FitPlot,ConvergencePlot,ChainsPlot,CornerPlot,ScopeRawPlot

import socket
is_turgon = socket.gethostname() == 'turgon'
if is_turgon:
    from .PyCCF import PyCCF



