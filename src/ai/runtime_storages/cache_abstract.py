from abc import ABC, abstractmethod


class CacheAbstract(ABC):
    @abstractmethod
    def invalidate_and_recalculate(self, storage):
        """
        Invalidate an element in the cache (e.g., mark as stale).
        """
        pass
