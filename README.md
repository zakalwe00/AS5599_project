# AS5599_project code repository
## Light curve analysis and reporting
This code respository can be used to generate light curve reverberation analysis for light curves using CSV format observations directly downloaded from [AVA]. Configuration files cover datasets from three Active Galactic Nuclei (AGN), Fairall 9, NGC 6814 and NGC 1365.
## Code organisation
```
project
│   README.md
│
└───code
│   │   atmos.py
│   │   get_args.py
│   │   launcher.py
│   │   launcher.sh
│   │   testbed.py
│   │   test_calibration.py
│   │   transmission.py
│   │
│   └───AGNLCLib
│       │   __init__.py
│       │   AGNLCModel.py
│       │   PyCCF.py
│       │   PyROA.py
│       │   PyROA_Plot.py
│       │   Utils.py
│   
└───config
    │   global.json
    │   NGC_1365_May22-Mar23.json
    │   NGC_6814_Aug22-Dec22.json
    │   NGC_6814_Mar23-Jun23.json
    │   Fairall_9_May20-Feb21.json
    │   Fairall_9_May21-Feb22.json
    │   Fairall_9_May22-Feb23.json
```
[!NOTE]
The `AGNLCLib` library in this project contain functions directly copied from [PyROA (Git)] (heavily edited to fit the configuration framework). These functions form the core of the Running Optimal Average (ROA) calibration, analysis and plotting. Their inclusion here is for experimental and learning purposes only, and should not be taken for the authors own work.

Additionally, to run the `CCF=1` switch mentioned below for the cross-correlation function requires the [PyCCF (bitbucket)] library to be installed in the users `PYTHONPATH`. Example code from this library is also incorporated in the `AGNLCLib\PyCCF.py` module.

##Steps to run the calibration and analysis:
 - Set up git repository on linux in `$HOME/git/AS5599_project` and select an `AGN` to run the analysis for. Available `JSON` configurations include those in the `config`1 directory above. The observing year is encoded into the `AGN` name as we found the calibration could drift year-on-year.
 - Populate the LCO csv file from AVA (eg) 
```$HOME/git/AS5599_project/Fairall_9_Jun18-Feb19/LCO/AVA_Fairall_9_lco.csv```
 - Run the `launcher.sh` script with the variable `AGN` exported and desired options switched on (command line switches `CALIBRATE=1`, `CCF=1`, `ROA=1`, `ANALYSIS=1` are available).
###Practial considerations
The `CALIBRATE` pipeline must be run first, and `ANALYSIS` last, though `CCF` and `ROA` can be run concurretly on a large enough machine. You are advised to run this with `DRYRUN=1` the first time and try a command manually to verify all libraries are correctly installed and your machine is sufficiently provisioned, eg:
```
user@server:~/git/AS5599_project/Fairall_9_Jun18-Feb19$ AGN=Fairall_9_Jun18-Feb19 CALIBRATE=1 DRYRUN=1 $HOME/git/AS5599_project/code/launcher.sh
Running signal analysis for Fairall_9_Jun18-Feb19
--LOGDIR--
Writing parameters to /home/user/git/AS5599_project/Fairall_9_Jun18-Feb19/output/tmp/20230803_153420/used_params.json
--CALIBRATION--
Running python /home/user/git/AS5599_project/code/launcher.py Fairall_9_Jun18-Feb19 calibrate:u 2>&1|cat > /home/user/git/AS5599_project/Fairall_9_Jun18-Feb19/output/tmp/20230803_153420/calibrate_u.log
Running python /home/user/git/AS5599_project/code/launcher.py Fairall_9_Jun18-Feb19 calibrate_filt_plot:u,sig 2>&1|cat > /home/user/git/AS5599_project/Fairall_9_Jun18-Feb19/output/tmp/20230803_153420/calibrate_filt_plot_usig.log
Running python /home/user/git/AS5599_project/code/launcher.py Fairall_9_Jun18-Feb19 calibrate_filt_plot:u,A 2>&1|cat > /home/user/git/AS5599_project/Fairall_9_Jun18-Feb19/output/tmp/20230803_153420/calibrate_filt_plot_uA.log
Running python /home/user/git/AS5599_project/code/launcher.py Fairall_9_Jun18-Feb19 calibrate_filt_plot:u,B 2>&1|cat > /home/user/git/AS5599_project/Fairall_9_Jun18-Feb19/output/tmp/20230803_153420/calibrate_filt_plot_uB.log
Running python /home/user/git/AS5599_project/code/launcher.py Fairall_9_Jun18-Feb19 calibrate:B 2>&1|cat > /home/user/git/AS5599_project/Fairall_9_Jun18-Feb19/output/tmp/20230803_153420/calibrate_B.log
.
.
```
The commands take a significantly long time to run (~8h each for calibration of individual filter bands and ~2 days for ROA). You are advised to run them in the background and monitor through the coming days, eg:
```
user@server:~/git/AS5599_project/Fairall_9_Jun18-Feb19$ AGN=Fairall_9_Jun18-Feb19 CALIBRATE=1 $HOME/git/AS5599_project/code/launcher.sh 2>&1|cat > pipeline_runs.20230803.log &
user@server:~/git/AS5599_project$ disown
```
###Testing calibration
After calibrating the first time the script `test_calibration.py` outputs a calibration analysis graph and and the dataset SNR. You are advised to look at this and consider whether any outlier observations are influencial to the fit as well as (optionally) removing these from the dataset.
###Log files
The `launcher.sh` script sets up a temporary log file in the output directory to allow individual stages of the pipeline to be monitored.


[//]: # (References)

   [AVA]: <http://alymantara.com/ava/index.php>
   [PyROA (Git)]: <https://github.com/FergusDonnan/PyROA>
   [PyCCF (bitbucket)]: <https://bitbucket.org/cgrier/python_ccf_code/src/master/>
