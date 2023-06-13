#!/bin/bash
LAUNCH_SCRIPT="$HOME/git/AS5599_project/code/launcher.py"
ARGS_SCRIPT="$HOME/git/AS5599_project/code/get_args.py"
LOG_DIR=`$ARGS_SCRIPT $AGN tmp_dir`
CALIBRATE=${CALIBRATE:-0}
CCF=${CCF:-0}
echo "Running signal analysis for $AGN"
echo "--LOGDIR--"
echo "Create $LOG_DIR"
mkdir -p $LOG_DIR
FLTRS=`$ARGS_SCRIPT $AGN fltrs`
if [ $CALIBRATE -eq 1 ]
then
    echo "--CALIBRATION--"
    for fltr in $FLTRS
    do
	echo "Running $LAUNCH_SCRIPT $AGN calibrate:$fltr 2>&1|cat > $LOG_DIR/calibrate_$fltr.log"
	$LAUNCH_SCRIPT $AGN calibrate:$fltr 2>&1|cat > $LOG_DIR/calibrate_$fltr.log
    done
fi

if [ $CCF -eq 1 ]
then
    echo "--CCF--"
    for fltr1 in $FLTRS
    do
	for fltr2 in $FLTRS
	do
	    if [[ $fltr1 < $fltr2 ]]
	    then
		echo "Running $LAUNCH_SCRIPT $AGN ccf:$fltr1,$fltr2 2>&1|cat > $LOG_DIR/ccf_$fltr1""_$fltr2.log"
		$LAUNCH_SCRIPT $AGN ccf:$fltr1,$fltr2 2>&1|cat > "$LOG_DIR/ccf_$fltr1""_$fltr2.log"
	    fi
	done
    done
fi
