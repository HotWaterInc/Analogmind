import copy
import unittest
import torch
from src import runtime_storages as storage
from src.runtime_storages.general_cache.cache_nodes_indexes import validate_cache_nodes_indexes, \
    CacheNodesIndexes
from src.runtime_storages.other.cache_functions import cache_general_get
from src.runtime_storages.general_cache.cache_nodes_map import validate_cache_nodes_map, CacheNodesMap
from src.runtime_storages.types import CacheGeneralAlias, NodeAuthenticData, ConnectionAuthenticData
from src.navigation_core.networks.common import training_loop, Loss, Mutation
from src.navigation_core.networks.metric_generator.build_dataloaders import build_dataloader_walking, \
    build_dataloader_rotations
from src.navigation_core.networks.metric_generator.build_training_data import build_walking_data, build_rotations_data
from src.navigation_core.networks.metric_generator.losses import loss_rotations, loss_walking_distance
from src.navigation_core.networks.metric_generator.mutations import mutate_walking_data_rotations, \
    build_mutation_function
from src.navigation_core.networks.metric_generator.network_class import MetricNetwork
from src.navigation_core.networks.metric_generator.training_data_struct import MetricTrainingData
from src.navigation_core.networks.metric_generator.training_params import MetricTrainingParams, \
    create_metric_training_params
from src.utils.utils import set_testing


class TestsMetricNetworkDataPrep(unittest.TestCase):
    """
    Testing cache capabilities inside of storage
    """

    @classmethod
    def setUp(cls):
        storage_struct = storage.create_storage()
        # fill storage
        node1 = NodeAuthenticData(name="node1", datapoints_array=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                  params={"param1": 1})
        node2 = NodeAuthenticData(name="node2", datapoints_array=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                  params={"param1": 1})
        storage.crud.create_nodes(storage_struct, [node1, node2])

        connection1 = ConnectionAuthenticData(name="connection1", start="node1", end="node2", distance=1.0,
                                              direction=[1, 1])
        connections = [connection1]
        storage.crud.create_connections_authentic(storage_struct, connections)
        set_testing(True)

        cls.storage_struct = storage_struct

    @classmethod
    def tearDown(cls):
        cls.storage_struct = None

    def test_building_and_mutations(self):
        """Testing crud on data affecting the cache"""
        training_params: MetricTrainingParams = create_metric_training_params()
        training_data: MetricTrainingData = MetricTrainingData()
        storage_struct = self.storage_struct

        build_rotations_data(
            storage_struct=storage_struct,
            training_data=training_data
        )
        build_walking_data(
            storage_struct=storage_struct,
            training_data=training_data
        )
        build_dataloader_rotations(
            training_data_struct=training_data,
            training_params=training_params
        )
        build_dataloader_walking(
            training_data_struct=training_data,
            training_params=training_params
        )
        rotations_data = next(training_data.rotations_dataloader)
        self.assertEqual(rotations_data[0][0][0].item(), 1)
        self.assertEqual(rotations_data[0][0][1].item(), 2)

        old_start, end, distance = next(training_data.walking_dataloader)
        old_start = copy.deepcopy(old_start)

        # testing dataloader with mutated data
        build_dataloader_walking(
            training_data_struct=training_data,
            training_params=training_params
        )
        mutate_walking_data_rotations(storage_struct, training_data)
        start, end, distance = next(training_data.walking_dataloader)

        for index in range(len(start)):
            for i in range(len(start[index])):
                self.assertNotEqual(start[index][i].item(), old_start[index][i].item())
