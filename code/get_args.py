import os,argparse
from AGNLCLib import AGNLCModel

# Fast interogation of setup variables for launcher.sh

# setup global variables for use in the data pipeline (these can be overridden in environment)
HOMEDIR = os.environ['HOME']
TESTEXT = os.environ.get('TESTEXT','')
#json files for project configuration
PROJECTDIR = os.environ.get('PROJECTDIR','{}/git/AS5599_project{}'.format(HOMEDIR,TESTEXT))
CONFIGDIR = os.environ.get('CONFIGDIR','{}/git/AS5599_project{}/config'.format(HOMEDIR,TESTEXT))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("agn", type=str,
                        help="AGN name"),
    parser.add_argument("arg", type=str,
                        help="setup variable query")

    args=parser.parse_args()

    config = AGNLCModel(PROJECTDIR, CONFIGDIR, args.agn).config()

    if args.arg == 'tmp_dir':
        print(config.tmp_dir(),end='')
    elif args.arg == 'output_dir':
        print(config.output_dir(),end='')
    elif args.arg == 'fltrs':
        print(' '.join(config.fltrs()),end='')
    elif args.arg == 'delay_ref':
        print('{}'.format(config.roa_params()['delay_ref']))
    elif args.arg == 'calib_fltrs':
        print(' '.join(config.calib_fltrs()),end='')
    elif args.arg == 'scopes':
        print(' '.join(config.scopes()),end='')
    elif args.arg == 'periods':
        print(' '.join(config.observation_params()['periods']),end='')

    exit(0)


                
