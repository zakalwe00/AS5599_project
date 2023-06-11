import os,json
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

    def __init__(self, root_dir, agn_name):
        json_settings = '{}/settings/{}.json'.format(root_dir,agn_name)
        
        # manage lightcurve (LCO) data, check available scope and filters
        output_dir = '{}/{}/output'.format(root_dir,agn_name)
        PUtils.check_and_create_dir(output_dir)    
        lco_lc_file = PUtils.load_lco_lightcurves(root_dir,agn_name)
        fltrs, scopes = PUtils.write_scope_filter_data(agn_name,lco_lc_file, output_dir)

        self._output_dir = output_dir
        self._agn_name   = agn_name
        self._fltrs      = fltrs
        self._scopes     = scopes

    def output_dir(self): return self._output_dir
    def agn_name(self): return self._agn_name
    def fltrs(self): return self._fltrs
    def scopes(self): return self._scopes
        
