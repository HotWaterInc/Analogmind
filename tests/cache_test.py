import unittest
from src import runtime_storages as storage
from src.runtime_storages.functionalities.functionalities_types import FunctionalityAlias
from src.runtime_storages.functions.cache_functions import cache_general_get
from src.runtime_storages.general_cache.cache_nodes_map import validate_cache_nodes_map, CacheNodesMap
from src.runtime_storages.types import DataAlias, CacheGeneralAlias, NodeAuthenticData


class CacheTests(unittest.TestCase):
    """
    Testing cache capabilities inside of storage
    """

    @classmethod
    def setUpClass(cls):
        cls.storage_struct = storage.create_storage()

    @classmethod
    def tearDownClass(cls):
        cls.storage_struct = None

    def test_node_cache_map(self):
        storage_struct = self.storage_struct
        node = NodeAuthenticData(name="node1", datapoints_array=[[1, 2, 3], [4, 5, 6]], params={"param1": 1})
        cache = cache_general_get(storage_struct, CacheGeneralAlias.NODE_CACHE_MAP)
        cache = validate_cache_nodes_map(cache)

        self.assertTrue(isinstance(cache, CacheNodesMap))
        with self.assertRaises(KeyError):
            cache.read(node_name="node1")
            cache.read(node_name="node2")
