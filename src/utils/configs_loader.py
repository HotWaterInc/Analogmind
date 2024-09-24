import configparser
import os


def load_config_ini(debug_default=False):
    config = configparser.ConfigParser()

    # Load config.ini, handle if the file doesn't exist
    try:
        config.read('config.ini')
    except FileNotFoundError:
        config = {}

    # Get the debug value from config.ini
    # Fallback to environment variable DEBUG or the provided default
    debug = config.getboolean('DEFAULT', 'DEBUG',
                              fallback=os.getenv('DEBUG', str(debug_default)).lower() in ['true', '1', 't', 'yes'])

    return debug
