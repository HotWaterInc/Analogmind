import enum

class CollectedDataType(enum.Enum):
    Data8x8 = 1
    Data15x15 = 2

class Paths(enum.Enum):
    Data8x8 = 'data/data8x8.json'
    Data15x15 = 'data/data15x15.json'

class AIType(enum.Enum):
    Autoencoder = 1
    VariationalAutoencoder = 2

class AIPaths(enum.Enum):
    Autoencoder = 'models/autoencoders/'
    VariationalAutoencoder = 'models/variational_autoencoders/'

AI_MODELS_TRACKER_PATH = 'models/ai_models_tracker.json'

def get_model_path(ai_type: AIType):
    if ai_type == AIType.Autoencoder:
        return AIPaths.Autoencoder.value
    elif ai_type == AIType.VariationalAutoencoder:
        return AIPaths.VariationalAutoencoder.value
    else:
        return None

def get_file_path(data_sample: CollectedDataType):
    if data_sample == CollectedDataType.Data8x8:
        return Paths.Data8x8.value
    elif data_sample == CollectedDataType.Data15x15:
        return Paths.Data15x15.value
    else:
        return None

DATA_NAME_FIELD = "name"
DATA_SENSORS_FIELD = "data"
DATA_PARAMS_FIELD = "params"
