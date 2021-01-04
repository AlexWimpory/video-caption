import logging
import os

vosk_log_level = -1

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)


def init():
    print('Initialising logging framework')
