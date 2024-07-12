class GlobalDataBuffer:
    """
    Simple way of sharing data between different parts of the application
    Used only for really simple stuff or as placeholder
    """
    __instance = None

    buffer: any = None
    empty_buffer: bool = True

    @staticmethod
    def get_instance():
        if GlobalDataBuffer.__instance is None:
            GlobalDataBuffer()
        return GlobalDataBuffer.__instance

    def __init__(self):
        if GlobalDataBuffer.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            GlobalDataBuffer.__instance = self


def empty_global_data_buffer():
    global_data_buffer: GlobalDataBuffer = GlobalDataBuffer.get_instance()
    global_data_buffer.buffer = None
    global_data_buffer.empty_buffer = True
