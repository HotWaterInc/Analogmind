import configparser
import os
from enum import Enum


class ConfigParameters(Enum):
    SHOW_NODE_INDEXES = 'NODE_INDEXES'


def load_config_ini_visualization(parameter: ConfigParameters) -> any:
    config = configparser.ConfigParser()

    # Load config.ini, handle if the file doesn't exist
    try:
        config.read('/src/visualizations/config.ini')
    except FileNotFoundError:
        config = {}

    param_value = config.get('DEFAULT', parameter.value, fallback=None)
    return param_value
