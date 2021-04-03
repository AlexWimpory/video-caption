import logging.config
import os
import sys
import traceback
import yaml
from vosk import SetLogLevel


def init():
    """Initialise logging config for the application"""
    config_file_name = os.environ.get('CONFIG_FILE_NAME', 'logging_config.yaml')
    print(f'Configuring the logging system from config file: {config_file_name}', flush=True)
    try:
        with open(os.path.join(os.path.dirname(__file__), config_file_name), 'r') as fin:
            yml = yaml.load(fin, Loader=yaml.FullLoader)
            logging.config.dictConfig(yml)
    except (TypeError, FileNotFoundError, ValueError):
        print('Failed to initialise the logging framework', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    # Set Vosk log level to silence output
    SetLogLevel(-1)
    # Set TensorFlow C++ logging to silence non error
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Call init when the module is imported
init()


def get_logger(name):
    """If we use this function we are forced to load this module"""
    return logging.getLogger(name)
