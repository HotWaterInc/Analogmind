from typing import TYPE_CHECKING
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

if TYPE_CHECKING:
    from src.runtime_storages import StorageStruct


def train_metric_generator_network(storage_struct: 'StorageStruct', network: MetricNetwork):
    training_params: MetricTrainingParams = create_metric_training_params()
    training_data: MetricTrainingData = MetricTrainingData()

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

    training_loop(
        network=network,
        training_params=training_params,
        training_data=training_data,
        losses=[
            Loss(name="rotation_loss", loss_function=loss_rotations, loss_scaling_factor=1),
            Loss(name="walking_loss", loss_function=loss_walking_distance, loss_scaling_factor=1)
        ],

        mutations=[
            Mutation(name="rotation_permutations",
                     mutation_function=build_mutation_function(storage_struct, mutate_walking_data_rotations),
                     epochs_rate=5),
        ]
    )
