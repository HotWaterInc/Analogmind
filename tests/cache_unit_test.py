import unittest
from src import runtime_storages as storage
from src.runtime_storages.other.cache_functions import cache_general_get
from src.runtime_storages.general_cache.cache_nodes_map import validate_cache_nodes_map, CacheNodesMap
from src.runtime_storages.types import CacheGeneralAlias, NodeAuthenticData


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
        """Testing crud on data affecting the cache"""

        storage_struct = self.storage_struct
        cache = cache_general_get(storage_struct, CacheGeneralAlias.NODE_CACHE_MAP)
        cache = validate_cache_nodes_map(cache)
        self.assertTrue(isinstance(cache, CacheNodesMap))

        # create node
        node = NodeAuthenticData(name="node1", datapoints_array=[[1, 2, 3], [4, 5, 6]], params={"param1": 1})
        with self.assertRaises(KeyError):
            cache.read(node_name="node1")
            cache.read(node_name="node2")
        storage.crud.create_nodes(storage_struct, [node])
        read_node = cache.read(node_name="node1")
        with self.assertRaises(KeyError):
            cache.read(node_name="node2")
        self.assertEqual(read_node, node)

        # update node
        new_params = {"param1": 2}
        self.assertEqual(read_node["params"]["param1"], 1)
        updated_node = NodeAuthenticData(name="node1", datapoints_array=[[1, 2, 3], [4, 5, 6]], params=new_params)
        storage.crud.update_nodes_by_name(storage_struct, names=["node1"], updated_nodes=[updated_node])
        read_node = cache.read(node_name="node1")
        self.assertEqual(read_node["params"]["param1"], 2)

        # create other node
        node2 = NodeAuthenticData(name="node2", datapoints_array=[[1, 2, 3], [4, 5, 6]], params={"param1": 1})
        with self.assertRaises(KeyError):
            cache.read(node_name="node2")
        storage.crud.create_nodes(storage_struct, [node2])
        read_node2 = cache.read(node_name="node2")
        self.assertEqual(read_node2, node2)
        read_node1 = cache.read(node_name="node1")
        self.assertNotEqual(read_node1, node2)

        # delete nodes
        storage.crud.delete_nodes(storage_struct, ["node1"])
        with self.assertRaises(KeyError):
            cache.read(node_name="node1")
        read_node2 = cache.read(node_name="node2")
        self.assertEqual(read_node2, node2)
        storage.crud.delete_nodes(storage_struct, ["node2"])
        with self.assertRaises(KeyError):
            cache.read(node_name="node2")
