import enum


class CollectedDataType(enum.Enum):
    """
    Enum for different types of data formats that might be collected from the environment

    Data8x8: 8x8 grid data
    Data15x15: 15x15 grid data
    Other: Other types of data
    """
    Data8x8 = 1
    Data15x15 = 2
    Other = 3


class Paths(enum.Enum):
    """
    Enum for different paths to data files
    """
    Data8x8 = 'data/data8x8.json'
    Data15x15 = 'data/data15x15.json'
    Other = 'data/other/'


class AIType(enum.Enum):
    """
    Enum for different types of AI models
    """
    Autoencoder = 1
    VariationalAutoencoder = 2
    ManuallySaved = 3
    Others = 4


class AIPaths(enum.Enum):
    """
    Enum for different paths to AI models
    """
    Autoencoder = 'models/autoencoders/'
    VariationalAutoencoder = 'models/variational_autoencoders/'
    ManuallySaved = 'models/manually_saved/'
    Others = 'models/others/'


AI_MODELS_TRACKER_PATH = 'models/ai_models_tracker.json'


def get_model_path(ai_type: AIType):
    """
    Get the path to the AI model based on the AI type ( up to the folder )
    """
    if ai_type == AIType.Autoencoder:
        return AIPaths.Autoencoder.value
    elif ai_type == AIType.VariationalAutoencoder:
        return AIPaths.VariationalAutoencoder.value
    elif ai_type == AIType.ManuallySaved:
        return AIPaths.ManuallySaved.value
    else:
        return AIPaths.Others.value


def get_data_file_path(data_sample: CollectedDataType):
    """
    Get the path to the data file based on the data sample type
    """
    if data_sample == CollectedDataType.Data8x8:
        return Paths.Data8x8.value
    elif data_sample == CollectedDataType.Data15x15:
        return Paths.Data15x15.value
    elif data_sample == CollectedDataType.Other:
        return Paths.Other.value


DATA_NAME_FIELD = "name"
DATA_SENSORS_FIELD = "data"
DATA_PARAMS_FIELD = "params"
