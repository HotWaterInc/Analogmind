import torch


def _train_autoencoder_with_distance_constraint(manifold_network: BaseAutoencoderModel, storage: StorageSuperset2,
                                                epochs: int, stop_at_threshold: bool = False) -> BaseAutoencoderModel:
    # PARAMETERS
    optimizer = optim.Adam(manifold_network.parameters(), lr=0.0002)

    num_epochs = epochs

    scale_reconstruction_loss = 1
    scale_non_adjacent_distance_loss = 10
    scale_adjacent_distance_loss = 1

    non_adjacent_sample_size = 1000
    adjacent_sample_size = 100
    permutation_sample_size = 100

    epoch_average_loss = 0

    reconstruction_average_loss = 0
    non_adjacent_average_loss = 0
    adjacent_average_loss = 0
    permutation_average_loss = 0

    epoch_print_rate = 1000
    DISTANCE_SCALING_FACTOR = 1
    EMBEDDING_SCALING_FACTOR = 0.1

    storage.build_permuted_data_random_rotations_rotation0()
    train_data = array_to_tensor(np.array(storage.get_pure_permuted_raw_env_data())).to(get_device())
    manifold_network = manifold_network.to(get_device())

    print("STARTED TRAINING")
    pretty_display_reset()
    pretty_display_set(epoch_print_rate, "Epoch batch")
    pretty_display_start(0)

    if stop_at_threshold:
        num_epochs = int(1e7)

    SHUFFLE_RATE = 2
    for epoch in range(num_epochs):
        if epoch % SHUFFLE_RATE == 0:
            storage.build_permuted_data_random_rotations()
            # storage.build_permuted_data_random_rotations_rotation0()

        reconstruction_loss = torch.tensor(0.0)
        adjacent_distance_loss = torch.tensor(0.0)
        non_adjacent_distance_loss = torch.tensor(0.0)
        permutation_adjustion_loss = torch.tensor(0.0)

        epoch_loss = 0.0
        optimizer.zero_grad()
        accumulated_loss = torch.tensor(0.0, device=get_device())

        # NON-ADJACENT DISTANCE LOSS
        non_adjacent_distance_loss = non_adjacent_distance_handling(manifold_network, storage, non_adjacent_sample_size,
                                                                    distance_scaling_factor=DISTANCE_SCALING_FACTOR,
                                                                    embedding_scaling_factor=EMBEDDING_SCALING_FACTOR)
        # PERMUTATION ADJUST LOSS
        permutation_adjustion_loss = permutation_adjustion_handling(manifold_network, storage, permutation_sample_size)

        accumulated_loss = reconstruction_loss + non_adjacent_distance_loss + adjacent_distance_loss + permutation_adjustion_loss
        accumulated_loss.backward()
        optimizer.step()

        epoch_loss += reconstruction_loss.item() + non_adjacent_distance_loss.item() + adjacent_distance_loss.item() + \
                      permutation_adjustion_loss.item()

        epoch_average_loss += epoch_loss

        reconstruction_average_loss += reconstruction_loss.item()
        non_adjacent_average_loss += non_adjacent_distance_loss.item()
        adjacent_average_loss += adjacent_distance_loss.item()
        permutation_average_loss += permutation_adjustion_loss.item()

        pretty_display(epoch % epoch_print_rate)

        if epoch % epoch_print_rate == 0 and epoch != 0:
            epoch_average_loss /= epoch_print_rate

            reconstruction_average_loss /= epoch_print_rate
            non_adjacent_average_loss /= epoch_print_rate
            adjacent_average_loss /= epoch_print_rate
            permutation_average_loss /= epoch_print_rate

            # Print average loss for this epoch
            print("")
            print(f"EPOCH:{epoch}/{num_epochs}")
            print(
                f"RECONSTRUCTION LOSS:{reconstruction_average_loss} | NON-ADJACENT LOSS:{non_adjacent_average_loss} | ADJACENT LOSS:{adjacent_average_loss} | PERMUTATION LOSS:{permutation_average_loss}")
            print(f"AVERAGE LOSS:{epoch_average_loss}")
            print("--------------------------------------------------")

            if non_adjacent_average_loss < THRESHOLD_MANIFOLD_NON_ADJACENT_LOSS and permutation_average_loss < THRESHOLD_MANIFOLD_PERMUTATION_LOSS and stop_at_threshold:
                print(f"Stopping at epoch {epoch} with loss {epoch_average_loss} because of threshold")
                break

            epoch_average_loss = 0
            reconstruction_average_loss = 0
            non_adjacent_average_loss = 0
            adjacent_average_loss = 0
            permutation_average_loss = 0

            pretty_display_reset()
            pretty_display_start(epoch)

    return manifold_network


def train_manifold_network_until_thresholds(manifold_network: BaseAutoencoderModel, storage: StorageSuperset2):
    manifold_network = _train_autoencoder_with_distance_constraint(
        manifold_network=manifold_network,
        storage=storage,
        epochs=-1,
        stop_at_threshold=True
    )

    return manifold_network


def train_manifold_network(manifold_network: BaseAutoencoderModel, storage: StorageSuperset2):
    manifold_network = _train_autoencoder_with_distance_constraint(
        manifold_network=manifold_network,
        storage=storage,
        epochs=5001,
        stop_at_threshold=False
    )

    return manifold_network
