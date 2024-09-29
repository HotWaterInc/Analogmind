import unittest
from src import runtime_storages as storage
from src.runtime_storages.functions.functionalities.functionalities_types import FunctionalityAlias
from src.runtime_storages.functions.functionalities.get_walk_distance import \
    validate_cache_get_walk_distance
from src.runtime_storages.other import cache_specialized_get
from src.runtime_storages.types import NodeAuthenticData, ConnectionAuthenticData


class TestsCacheSpecialized(unittest.TestCase):
    """
    Testing cache capabilities inside of storage
    """

    @classmethod
    def setUp(cls):
        cls.storage_struct = storage.create_storage()

    @classmethod
    def tearDown(cls):
        cls.storage_struct = None

    def test_get_walk_cache(self):
        """Testing crud on data affecting the cache"""

        storage_struct = self.storage_struct
        cache = cache_specialized_get(storage_struct, FunctionalityAlias.GET_WALK_DISTANCE)
        cache = validate_cache_get_walk_distance(cache)

        node1 = NodeAuthenticData(name="node1", datapoints_array=[[1, 2, 3], [4, 5, 6]], params={"param1": 1})
        node2 = NodeAuthenticData(name="node2", datapoints_array=[[1, 2, 3], [4, 5, 6]], params={"param1": 1})
        node3 = NodeAuthenticData(name="node3", datapoints_array=[[1, 2, 3], [4, 5, 6]], params={"param1": 1})
        node4 = NodeAuthenticData(name="node4", datapoints_array=[[1, 2, 3], [4, 5, 6]], params={"param1": 1})
        node5 = NodeAuthenticData(name="node5", datapoints_array=[[1, 2, 3], [4, 5, 6]], params={"param1": 1})

        storage.crud.create_nodes(storage_struct, [node1, node2, node3, node4, node5])

        connection1 = ConnectionAuthenticData(name="connection1", start="node1", end="node2", distance=1.0,
                                              direction=[1, 1])
        connection2 = ConnectionAuthenticData(name="connection2", start="node2", end="node3", distance=1.0,
                                              direction=[1, 1])
        connection3 = ConnectionAuthenticData(name="connection3", start="node3", end="node4", distance=1.0,
                                              direction=[1, 1])

        connection4 = ConnectionAuthenticData(name="connection4", start="node4", end="node5", distance=1.0,
                                              direction=[1, 1])
        connection5 = ConnectionAuthenticData(name="connection5", start="node5", end="node1", distance=1.0,
                                              direction=[1, 1])

        connections = [connection1, connection2, connection3, connection4, connection5]
        storage.crud.create_connections_authentic(storage_struct, connections)

        d12 = storage.get_walk_distance(storage=storage_struct, start_node="node1", end_node="node2")
        d23 = storage.get_walk_distance(storage=storage_struct, start_node="node2", end_node="node3")
        d13 = storage.get_walk_distance(storage=storage_struct, start_node="node1", end_node="node3")
        d15 = storage.get_walk_distance(storage=storage_struct, start_node="node1", end_node="node5")
        d14 = storage.get_walk_distance(storage=storage_struct, start_node="node1", end_node="node4")

        self.assertEqual(d12, 1.0)
        self.assertEqual(d23, 1.0)
        self.assertEqual(d13, 2.0)
        # has direct connection
        self.assertEqual(d15, 1.0)
        # uses node 4 to get to 5, 1->5->4
        self.assertEqual(d14, 2.0)
