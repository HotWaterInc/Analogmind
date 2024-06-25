import torch
from parameters import AI_MODELS_TRACKER_PATH
from parameters import AIPaths, AIType, get_model_path
import json

def get_current_track_number(model_type: AIType):
    with open(AI_MODELS_TRACKER_PATH, 'r') as file:
        data = json.load(file)

    if AIType.Autoencoder == model_type:
        return data['autoencoder']['track_number']
    elif AIType.VariationalAutoencoder == model_type:
        return data['variational_autoencoder']['track_number']

def update_tracker(model_type: AIType, complete_name: str):
    with open(AI_MODELS_TRACKER_PATH, 'r') as file:
        data = json.load(file)

    if AIType.Autoencoder == model_type:
        data['autoencoder']['track_number'] += 1
        data['autoencoder']['models'].append(complete_name)
    elif AIType.VariationalAutoencoder == model_type:
        data['variational_autoencoder']['track_number'] += 1
        data['variational_autoencoder']['models'].append(complete_name)

    with open(AI_MODELS_TRACKER_PATH, 'w') as file:
        json.dump(data, file, indent=4)


def save_ai(name, model_type: AIType):
    model_path = get_model_path(model_type)
    last_track_number = get_current_track_number(model_type)
    name = name + "_" + str(last_track_number) + ".pth"

    if AIType.Autoencoder == model_type:
        name = "autoencoder_" + name
    elif AIType.VariationalAutoencoder == model_type:
        name = "variational_autoencoder_" + name

    torch.save(name, model_path + name)
    update_tracker(model_type, name)

def get_latest_model_name(model_type: AIType):
    with open(AI_MODELS_TRACKER_PATH, 'r') as file:
        data = json.load(file)
        if AIType.Autoencoder == model_type:
            return data['autoencoder']['models'][-1]
        if AIType.VariationalAutoencoder == model_type:
            return data['variational_autoencoder']['models'][-1]

def load_ai(model_type: AIType, latest: bool = True, complete_name: str = ""):
    if latest:
        last_track_number = get_current_track_number(model_type)
        complete_name = get_latest_model_name(model_type)

    model_path = get_model_path(model_type)
    model = torch.load(model_path + complete_name)
    return model

