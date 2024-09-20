from abc import ABC, abstractmethod


class CacheAbstract(ABC):
    @abstractmethod
    def read(self, *args):
        pass

    pass
