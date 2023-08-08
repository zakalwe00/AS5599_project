#!/bin/bash
if [ -z $HOME ]
then
    echo "HOME environment variable not defined- ERROR"
    exit
fi
   
if [ -z $AGN ]
then
    echo "AGN not defined- please set as environment variable"
    exit
fi
SCRIPT=${BASH_SOURCE[0]}
SCRIPTDIR=`dirname $SCRIPT`
LAUNCH_SCRIPT="python $SCRIPTDIR/launcher.py"
ARGS_SCRIPT="python $SCRIPTDIR/get_args.py"
#This builds a temporary log dir and populates it with a backup of the config
LOG_DIR=`$ARGS_SCRIPT $AGN tmp_dir`
DRYRUN=${DRYRUN:-0}
RAW_PLOT=${RAW_PLOT:-0}
CALIBRATE=${CALIBRATE:-0}
CCF=${CCF:-0}
ROA=${ROA:-0}
ANALYSIS=${ANALYSIS:-0}
echo "Running data pipeline for $AGN"
echo "--LOGDIR--"
echo "Writing parameters to $LOG_DIR/used_params.json"
FLTRS=`$ARGS_SCRIPT $AGN fltrs`
PERIODS=`$ARGS_SCRIPT $AGN periods`
DELAY_REF=`$ARGS_SCRIPT $AGN delay_ref`

if [ $RAW_PLOT -eq 1 ]
then
    for fltr in $FLTRS
    do
	echo "Running $LAUNCH_SCRIPT $AGN raw_plot:$fltr 2>&1|cat > $LOG_DIR/raw_plot_$fltr.log"
	if [ $DRYRUN -ne 1 ]
	then
	    $LAUNCH_SCRIPT $AGN raw_plot:$fltr 2>&1|cat > "$LOG_DIR/raw_plot_$fltr.log"
	fi
    done
fi

if [ $CALIBRATE -eq 1 ]
then
    echo "--CALIBRATION--"
    for fltr in $FLTRS
    do
	echo "Running $LAUNCH_SCRIPT $AGN calibrate:$fltr 2>&1|cat > $LOG_DIR/calibrate_$fltr.log"
	if [ $DRYRUN -ne 1 ]
	then
	    $LAUNCH_SCRIPT $AGN calibrate:$fltr 2>&1|cat > "$LOG_DIR/calibrate_$fltr.log"
	fi
	for select in sig A B
	do
	    echo "Running $LAUNCH_SCRIPT $AGN calibrate_filt_plot:$fltr,$select 2>&1|cat > $LOG_DIR/calibrate_filt_plot_$fltr$select.log"
	    if [ $DRYRUN -ne 1 ]
	    then
		$LAUNCH_SCRIPT $AGN calibrate_filt_plot:$fltr,$select 2>&1|cat > "$LOG_DIR/calibrate_filt_plot_$fltr$select.log"
	    fi
	done
    done
    for period in $PERIODS
    do
	echo "Running $LAUNCH_SCRIPT $AGN calibrate_plot:$period 2>&1|cat > $LOG_DIR/calibrate_plot_$period.log"
	if [ $DRYRUN -ne 1 ]
	then
	    $LAUNCH_SCRIPT $AGN calibrate_plot:$period 2>&1|cat > "$LOG_DIR/calibrate_plot_$period.log"
	fi
    done
fi

if [ $CCF -eq 1 ]
then
    echo "--CCF--"
    for fltr in $FLTRS
    do
#       We want to include the delay ref to get error bar estimates?
#	if [[ $fltr != $DELAY_REF ]]
#	then
	echo "Running $LAUNCH_SCRIPT $AGN ccf:$DELAY_REF,$fltr 2>&1|cat > $LOG_DIR/ccf_$fltr.log"
	if [ $DRYRUN -ne 1 ]
	then
	    $LAUNCH_SCRIPT $AGN ccf:$DELAY_REF,$fltr 2>&1|cat > "$LOG_DIR/ccf_$fltr.log"
	fi
#	fi
    done
fi

if [ $ROA -eq 1 ]
then
    echo "--ROA--"
    for period in $PERIODS
    do
	echo "Running $LAUNCH_SCRIPT $AGN roa:$period 2>&1|cat > $LOG_DIR/roa_$period.log"
	if [ $DRYRUN -ne 1 ]
	then
	    $LAUNCH_SCRIPT $AGN roa:$period 2>&1|cat > "$LOG_DIR/roa_$period.log"
	fi
    done
fi

if [ $ANALYSIS -eq 1 ]
then
    echo "--ANALYSIS--"
    for period in $PERIODS
    do
	echo "Running $LAUNCH_SCRIPT $AGN roa_plot:$period 2>&1|cat > $LOG_DIR/roa_plot_$period.log"
	if [ $DRYRUN -ne 1 ]
	then
	    $LAUNCH_SCRIPT $AGN roa_plot:$period 2>&1|cat > "$LOG_DIR/roa_plot_$period.log"
	fi
	echo "Running $LAUNCH_SCRIPT $AGN roa_conv_plot:$period 2>&1|cat > $LOG_DIR/roa_conv_plot_$period.log"
	if [ $DRYRUN -ne 1 ]
	then
	    $LAUNCH_SCRIPT $AGN roa_conv_plot:$period 2>&1|cat > "$LOG_DIR/roa_conv_plot_$period.log"
	fi
	echo "Running $LAUNCH_SCRIPT $AGN roa_chains_plot:$period,tau 2>&1|cat > $LOG_DIR/roa_chains_plot_tau_$period.log"
	if [ $DRYRUN -ne 1 ]
	then
	    $LAUNCH_SCRIPT $AGN roa_chains_plot:$period,tau 2>&1|cat > "$LOG_DIR/roa_chains_plot_tau_$period.log"
	fi
	echo "Running $LAUNCH_SCRIPT $AGN roa_chains_plot:$period,delta 2>&1|cat > $LOG_DIR/roa_chains_plot_delta_$period.log"
	if [ $DRYRUN -ne 1 ]
	then
	    $LAUNCH_SCRIPT $AGN roa_chains_plot:$period,delta 2>&1|cat > "$LOG_DIR/roa_chains_plot_delta_$period.log"
	fi
	echo "Running $LAUNCH_SCRIPT $AGN roa_corner_plot:$period,tau 2>&1|cat > $LOG_DIR/roa_corner_plot_tau_$period.log"
	if [ $DRYRUN -ne 1 ]
	then
	    $LAUNCH_SCRIPT $AGN roa_corner_plot:$period,tau 2>&1|cat > "$LOG_DIR/roa_corner_plot_tau_$period.log"
	fi
	echo "Running $LAUNCH_SCRIPT $AGN roa_fluxflux:$period 2>&1|cat > $LOG_DIR/roa_fluxflux_$period.log"
	if [ $DRYRUN -ne 1 ]
	then
	    $LAUNCH_SCRIPT $AGN roa_fluxflux:$period 2>&1|cat > "$LOG_DIR/roa_fluxflux_$period.log"
	fi
	echo "Running $LAUNCH_SCRIPT $AGN roa_lagspectrum:$period 2>&1|cat > $LOG_DIR/roa_lagspectrum_$period.log"
	if [ $DRYRUN -ne 1 ]
	then
	    $LAUNCH_SCRIPT $AGN roa_lagspectrum:$period 2>&1|cat > "$LOG_DIR/roa_lagspectrum_$period.log"
	fi
    done
fi
