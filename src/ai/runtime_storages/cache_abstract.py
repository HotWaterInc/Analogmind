from abc import ABC, abstractmethod


class CacheAbstract(ABC):
    @abstractmethod
    def read(*args):
        pass

    pass
