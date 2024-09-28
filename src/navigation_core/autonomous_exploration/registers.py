from typing import Callable, Dict, List


class ExplorationBuffers:
    """
    Buffer solution for passing data around exploration. Will need refactoring if complexity grows
    """
    __instance = None

    list_buffer: List[any]
    dict_buffer: Dict[str, any]
    value_buffer: any

    @staticmethod
    def get_instance():
        if ExplorationBuffers.__instance is None:
            ExplorationBuffers()
        return ExplorationBuffers.__instance

    def __init__(self):
        if ExplorationBuffers.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            ExplorationBuffers.__instance = self
            self.initialize_defaults()


def set_list_buffer(data: List[any]):
    exploration_buffers: ExplorationBuffers = ExplorationBuffers.get_instance()
    exploration_buffers.list_buffer = data


def get_list_buffer() -> List[any]:
    exploration_buffers: ExplorationBuffers = ExplorationBuffers.get_instance()
    return exploration_buffers.list_buffer


def set_dict_buffer(data: Dict[str, any]):
    exploration_buffers: ExplorationBuffers = ExplorationBuffers.get_instance()
    exploration_buffers.dict_buffer = data


def get_dict_buffer() -> Dict[str, any]:
    exploration_buffers: ExplorationBuffers = ExplorationBuffers.get_instance()
    return exploration_buffers.dict_buffer


def set_value_buffer(data: any):
    exploration_buffers: ExplorationBuffers = ExplorationBuffers.get_instance()
    exploration_buffers.value_buffer = data


def get_value_buffer() -> any:
    exploration_buffers: ExplorationBuffers = ExplorationBuffers.get_instance()
    return exploration_buffers.value_buffer


def reset_value_buffer():
    exploration_buffers: ExplorationBuffers = ExplorationBuffers.get_instance()
    exploration_buffers.value_buffer = None


def reset_dict_buffer():
    exploration_buffers: ExplorationBuffers = ExplorationBuffers.get_instance()
    exploration_buffers.dict_buffer = None


def reset_list_buffer():
    exploration_buffers: ExplorationBuffers = ExplorationBuffers.get_instance()
    exploration_buffers.list_buffer = None


def get_and_reset_value_buffer() -> any:
    value = get_value_buffer()
    reset_value_buffer()
    return value


def get_and_reset_dict_buffer() -> Dict[str, any]:
    value = get_dict_buffer()
    reset_dict_buffer()
    return value


def get_and_reset_list_buffer() -> List[any]:
    value = get_list_buffer()
    reset_list_buffer()
    return value
