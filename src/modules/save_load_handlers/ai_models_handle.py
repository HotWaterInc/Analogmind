import torch
from .parameters import AI_MODELS_TRACKER_PATH
from .parameters import AIType, get_model_path
import json
from src.utils import prefix_path_with_root


def get_current_track_number(model_type: AIType):
    absolute_path = prefix_path_with_root(AI_MODELS_TRACKER_PATH)
    with open(absolute_path, 'r') as file:
        data = json.load(file)

    if AIType.Autoencoder == model_type:
        return data['autoencoder']['track_number']
    elif AIType.VariationalAutoencoder == model_type:
        return data['variational_autoencoder']['track_number']


def update_tracker(model_type: AIType, complete_name: str):
    absolute_path = prefix_path_with_root(AI_MODELS_TRACKER_PATH)
    with open(absolute_path, 'r') as file:
        data = json.load(file)

    last_track_number = get_current_track_number(model_type)
    data['autoencoder']['track_number'] += 1
    if AIType.Autoencoder == model_type:
        data['autoencoder']['models'][last_track_number] = complete_name
    elif AIType.VariationalAutoencoder == model_type:
        data['variational_autoencoder']['models'][last_track_number] = complete_name

    with open(absolute_path, 'w') as file:
        json.dump(data, file, indent=4)


def save_ai(name: str, model_type: AIType, model) -> None:
    model_path = get_model_path(model_type)
    last_track_number = get_current_track_number(model_type)
    name = name + "_" + str(last_track_number) + ".pth"

    local_model_path = model_path + name
    absolute_model_path = prefix_path_with_root(local_model_path)

    torch.save(model, absolute_model_path)
    update_tracker(model_type, name)


def save_ai_manually(name: str, model) -> None:
    model_path = get_model_path(AIType.ManuallySaved)
    local_model_path = model_path + name + ".pth"
    absolute_model_path = prefix_path_with_root(local_model_path)
    torch.save(model, absolute_model_path)


def get_model_name_by_version(model_type: AIType, version: int):
    absolute_path = prefix_path_with_root(AI_MODELS_TRACKER_PATH)
    with open(absolute_path, 'r') as file:
        data = json.load(file)
        if AIType.Autoencoder == model_type:
            return data['autoencoder']['models'][version]
        if AIType.VariationalAutoencoder == model_type:
            return data['variational_autoencoder']['models'][version]


def get_latest_model_name(model_type: AIType):
    absolute_path = prefix_path_with_root(AI_MODELS_TRACKER_PATH)
    last_track_number = f"{get_current_track_number(model_type) - 1}"

    with open(absolute_path, 'r') as file:
        data = json.load(file)
        print(data)
        if AIType.Autoencoder == model_type:
            return data['autoencoder']['models'][last_track_number]
        if AIType.VariationalAutoencoder == model_type:
            return data['variational_autoencoder']['models'][last_track_number]


def load_latest_ai(model_type: AIType):
    complete_name = get_latest_model_name(model_type)
    model_path = get_model_path(model_type)
    absolute_model_path = prefix_path_with_root(model_path + complete_name)

    model = torch.load(absolute_model_path)

    return model


def load_ai_version(model_type: AIType, version: int):
    complete_name = get_model_name_by_version(model_type, version)
    model_path = get_model_path(model_type)
    absolute_model_path = prefix_path_with_root(model_path + complete_name)
    model = torch.load(absolute_model_path)
    return model


def load_manually_saved_ai(model_name: str):
    model_path = get_model_path(AIType.ManuallySaved)
    absolute_model_path = prefix_path_with_root(model_path + model_name)
    model = torch.load(absolute_model_path)
    return model
