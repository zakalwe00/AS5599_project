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
LAUNCH_SCRIPT="$HOME/git/AS5599_project/code/launcher.py"
ARGS_SCRIPT="$HOME/git/AS5599_project/code/get_args.py"
LOG_DIR=`$ARGS_SCRIPT $AGN tmp_dir`
DRYRUN=${DRYRUN:-0}
CALIBRATE=${CALIBRATE:-0}
CCF=${CCF:-0}
ROA=${ROA:-0}
ROAPLOT=${ROAPLOT:-0}
echo "Running signal analysis for $AGN"
echo "--LOGDIR--"
echo "Running mkdir -p $LOG_DIR"
if [ $DRYRUN -ne 1 ]
   then
       mkdir -p $LOG_DIR
fi
FLTRS=`$ARGS_SCRIPT $AGN fltrs`
PERIODS=`$ARGS_SCRIPT $AGN periods`
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
       done
       echo "Running $LAUNCH_SCRIPT $AGN calibrate_plot 2>&1|cat > $LOG_DIR/calibrate_plot.log"
       if [ $DRYRUN -ne 1 ]
       then
	   $LAUNCH_SCRIPT $AGN calibrate_plot 2>&1|cat > "$LOG_DIR/calibrate_plot.log"
       fi
fi

if [ $CCF -eq 1 ]
   then
       echo "--CCF--"
       for fltr in $FLTRS
       do
	   if [[ $fltr != 'g' ]]
	      then
		  echo "Running $LAUNCH_SCRIPT $AGN ccf:g,$fltr 2>&1|cat > $LOG_DIR/ccf_g_$fltr.log"
		  if [ $DRYRUN -ne 1 ]
		     then
			 $LAUNCH_SCRIPT $AGN ccf:g,$fltr 2>&1|cat > "$LOG_DIR/ccf_g_$fltr.log"
		  fi
	   fi
       done
fi

if [ $ROA -eq 1 ]
   then
       echo "--ROA--"
       echo "Running $LAUNCH_SCRIPT $AGN roa 2>&1|cat > $LOG_DIR/roa.log"
       if [ $DRYRUN -ne 1 ]
       then
	   $LAUNCH_SCRIPT $AGN roa 2>&1|cat > "$LOG_DIR/roa.log"
       fi
fi

if [ $ROAPLOT -eq 1 ]
   then
       for period in $PERIODS
       do
	   echo "Running $LAUNCH_SCRIPT $AGN roa_plot:$period 2>&1|cat > $LOG_DIR/roa_plot_$period.log"
	   if [ $DRYRUN -ne 1 ]
	   then
	       $LAUNCH_SCRIPT $AGN roa_plot:$period 2>&1|cat > "$LOG_DIR/roa_plot_$period.log"
	   fi
       done
       echo "Running $LAUNCH_SCRIPT $AGN roa_conv_plot 2>&1|cat > $LOG_DIR/roa_conv_plot.log"
       if [ $DRYRUN -ne 1 ]
       then
	   $LAUNCH_SCRIPT $AGN roa_conv_plot 2>&1|cat > "$LOG_DIR/roa_conv_plot.log"
       fi
       echo "Running $LAUNCH_SCRIPT $AGN roa_chains_plot:tau 2>&1|cat > $LOG_DIR/roa_chains_plot_tau.log"
       if [ $DRYRUN -ne 1 ]
       then
	   $LAUNCH_SCRIPT $AGN roa_chains_plot:tau 2>&1|cat > "$LOG_DIR/roa_chains_plot_tau.log"
       fi
       echo "Running $LAUNCH_SCRIPT $AGN roa_chains_plot:delta 2>&1|cat > $LOG_DIR/roa_chains_plot_delta.log"
       if [ $DRYRUN -ne 1 ]
       then
	   $LAUNCH_SCRIPT $AGN roa_chains_plot:delta 2>&1|cat > "$LOG_DIR/roa_chains_plot_delta.log"
       fi
       echo "Running $LAUNCH_SCRIPT $AGN roa_corner_plot 2>&1|cat > $LOG_DIR/roa_corner_plot.log"
       if [ $DRYRUN -ne 1 ]
       then
	   $LAUNCH_SCRIPT $AGN roa_corner_plot 2>&1|cat > "$LOG_DIR/roa_corner_plot.log"
       fi
fi