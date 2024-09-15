from src.ai.runtime_storages.new.cache_abstract import CacheAbstract
from src.ai.runtime_storages.new.types import NodeData


class CacheMapNodes(CacheAbstract):
    def __init__(self):
        self.map = {}

    def create(self, storage, node: NodeData):
        self.map[node["name"]] = node

    def update(self, storage, node: NodeData):
        self.map[node["name"]] = node

    def delete(self, storage, node: NodeData):
        self.map[node["name"]] = None

    def invalidate_and_recalculate(self, storage):
        nodes = storage
        pass
