from abc import ABC, abstractmethod


class CacheAbstract(ABC):
    @abstractmethod
    def create(self, storage, element):
        """
        Create a new element in the cache.
        """
        pass

    @abstractmethod
    def update(self, storage, element):
        """
        Update an existing element in the cache.
        """
        pass

    @abstractmethod
    def delete(self, storage, element):
        """
        Delete an element from the cache.
        """
        pass

    @abstractmethod
    def invalidate_and_recalculate(self, storage):
        """
        Invalidate an element in the cache (e.g., mark as stale).
        """
        pass
