from multiprocessing.forkserver import set_forkserver_preload

import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
from fontTools.misc.cython import returns
from pyglet.input.linux.evdev import get_devices
import torch
from src.ai.models.permutor_autoenc_pipelined import storage
from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2, distance_thetas_to_distance_percent, \
    distance_percent_to_distance_thetas
from src.ai.variants.exploration.evaluation_exploration import eval_distances_threshold_averages_seen_network
from src.ai.variants.exploration.utils import MAX_DISTANCE
from src.modules.pretty_display import pretty_display_start, set_pretty_display, pretty_display, pretty_display_reset
from src.modules.save_load_handlers.ai_models_handle import load_ai_version, load_other_ai, load_manually_saved_ai
from src.modules.save_load_handlers.data_handle import read_other_data_from_file
from src.utils import get_device

# data from abstract block the last one
data = [
    [0.8362798574639949, 25.225257873535156, 0.007493830751627684],
    [0.5995681779414248, 17.992618560791016, 0.007893074303865433],
    [0.8471717653463199, 16.97913360595703, 0.006824419368058443],
    [0.08614522621712693, 15.495709419250488, 0.004345925524830818],
    [0.9826830618261414, 15.507229804992676, 0.008258088491857052],
    [0.23195258136093247, 15.249159812927246, 0.005343906115740538],
    [0.7335236874157507, 13.659500122070312, 0.00387671054340899],
    [0.7228699744767382, 14.190435409545898, 0.005616890266537666],
    [0.1158317745698477, 9.256025314331055, 0.004214365966618061],
    [0.9548581046417314, 15.82748031616211, 0.005577942356467247],
    [0.6197096094139576, 6.664989948272705, 0.004659573081880808],
    [0.21440149253211838, 11.291770935058594, 0.003782890737056732],
    [0.6719598202273704, 8.740020751953125, 0.004929320886731148],
    [0.6460123837822305, 19.777584075927734, 0.0054673245176672935],
    [0.6484597134749391, 12.620597839355469, 0.004020335618406534],
    [0.28848743473503335, 16.678464889526367, 0.0043764482252299786],
    [0.3690880653719381, 10.964065551757812, 0.0027209597174078226],
    [1.1113955191559843, 17.635211944580078, 0.0064543201588094234],
    [0.08981091247727079, 14.275008201599121, 0.004921969957649708],
    [0.13232157798333594, 20.76603889465332, 0.004922182764858007],
    [0.2859108252585063, 4.177954196929932, 0.0022728266194462776],
    [0.15713688300332326, 30.594783782958984, 0.0046976529993116856],
    [0.42206397619318303, 25.07473373413086, 0.0036707737017422915],
    [0.8497935043291399, 14.707901954650879, 0.0054719103500247],
    [0.20082081565415502, 5.823964595794678, 0.0046500456519424915],
    [0.5248580760548511, 17.978036880493164, 0.007429017219692469],
    [1.1818244370463828, 13.312263488769531, 0.004827719647437334],
    [1.277349208321671, 21.379940032958984, 0.005665302742272615],
    [0.4274225075964063, 20.682302474975586, 0.009519779123365879],
    [0.007280109889280205, 12.095253944396973, 0.004843092989176512],
    [0.8263758224924056, 15.364405632019043, 0.008709642104804516],
    [0.9359700849920366, 23.278636932373047, 0.0043677520006895065],
    [0.821194861162684, 22.815814971923828, 0.006177688017487526],
    [0.7786578195844441, 11.81525993347168, 0.0040789940394461155],
    [0.5498672567083805, 18.951534271240234, 0.006654929369688034],
    [1.1205802068571442, 12.506682395935059, 0.006011776626110077],
    [0.5723495435483461, 16.094318389892578, 0.004414999391883612],
    [1.1973512433701317, 20.97266960144043, 0.006670759059488773],
    [0.08900561780022685, 12.205307960510254, 0.0029466645792126656],
    [1.0330755054689855, 20.87548828125, 0.007094112690538168],
    [1.2218968859932493, 14.211782455444336, 0.009150250814855099],
    [0.7771550681813759, 13.333024024963379, 0.0035799602046608925],
    [0.08876936408468829, 7.789279460906982, 0.0032707140780985355],
    [0.6463512976702379, 19.68724250793457, 0.004349036607891321],
    [0.4117681386411535, 23.851972579956055, 0.00637347437441349],
    [0.72366083215827, 14.172133445739746, 0.007174250669777393],
    [1.0008896043020925, 22.619455337524414, 0.005527519155293703],
    [0.8940123041658877, 11.242576599121094, 0.007104638032615185],
    [0.6360793975597702, 16.886741638183594, 0.005357126705348492],
    [1.1877394495427016, 9.956257820129395, 0.003317589871585369],
    [0.6346534487419098, 25.93065071105957, 0.006523038726300001],
    [0.5336384543864887, 12.154642105102539, 0.0042877523228526115],
    [0.36788313361718555, 14.022916793823242, 0.006119241006672382],
    [0.625182373391957, 17.24979019165039, 0.005181158427149057],
    [1.4662854428793868, 12.954696655273438, 0.004267213400453329],
    [0.4964070910049533, 13.012557029724121, 0.005537726450711489],
    [0.5526608363182611, 19.24344825744629, 0.004485837183892727],
    [0.1280039061903964, 34.27313995361328, 0.007568240165710449],
    [0.6689469336202987, 11.957494735717773, 0.0031935039442032576],
    [0.7188393422733623, 19.313045501708984, 0.006738135125488043],
    [0.47448182262337496, 23.140161514282227, 0.007602407597005367],
    [0.5263800908089136, 13.046121597290039, 0.007493054959923029],
    [1.1250284440848597, 21.067697525024414, 0.008436008356511593],
    [0.18961540021844198, 12.6217679977417, 0.0026732818223536015],
    [0.2989314302645339, 10.596657752990723, 0.004984479397535324],
    [0.9288756644460008, 24.76005744934082, 0.005925923585891724],
    [0.3556810368855782, 18.837839126586914, 0.009590604342520237],
    [0.6512572456410752, 14.596170425415039, 0.009111430495977402],
    [0.1472990156111031, 18.1770076751709, 0.007539082784205675],
    [0.5900932129757127, 29.601123809814453, 0.006312158890068531],
    [0.825717263959038, 18.31293296813965, 0.004874975420534611],
    [0.8765483443598533, 15.241477966308594, 0.00934858899563551],
    [0.3716019375622253, 14.403572082519531, 0.006136896088719368],
    [0.5788307179132777, 16.644168853759766, 0.0037167826667428017],
    [0.36503424496887943, 23.612184524536133, 0.003946827724575996],
    [1.4008069103199055, 23.73312759399414, 0.007435070350766182],
    [0.35079481182024347, 19.26349639892578, 0.007537239231169224],
    [0.28431320757221257, 10.963915824890137, 0.005894344765692949],
    [0.9151535390304733, 14.694469451904297, 0.011626320891082287],
    [0.1632207094703365, 15.759420394897461, 0.007718371693044901],
    [0.6302063154237664, 18.89801788330078, 0.006495247129350901],
    [0.8689223210391132, 22.700130462646484, 0.009302135556936264],
    [0.7179651802141942, 14.173707008361816, 0.004137718118727207],
    [0.14796283317103642, 18.645896911621094, 0.006376306526362896],
    [0.8197658202194087, 6.692897796630859, 0.0074138096533715725],
    [0.14940214188558346, 9.186388969421387, 0.004447314888238907],
    [0.829472724084403, 15.624059677124023, 0.0044738114811480045],
    [0.2810071173475862, 16.946474075317383, 0.004617383237928152],
    [0.936318321939713, 13.663219451904297, 0.007699486333876848],
    [0.47412340165826, 19.575620651245117, 0.004071758594363928],
    [0.2115892246783843, 16.0473575592041, 0.0044526406563818455],
    [1.1115507185909244, 14.062698364257812, 0.00659079710021615],
    [0.9478929264426448, 31.999130249023438, 0.006096548866480589],
    [0.3014050430898593, 13.66552448272705, 0.003111080266535282]
]


def eval_data_changes(storage: StorageSuperset2, seen_network: any) -> any:
    connections = storage.get_all_connections_data()
    SAMPLES = min(200, len(connections))
    seen_network.eval()
    seen_network = seen_network.to(get_device())

    sampled_connections = np.random.choice(np.array(connections), SAMPLES, replace=False)
    connections_distances_data = []

    start_data_arr = []
    end_data_arr = []

    same_position_difference = 0
    different_position_difference = 0

    for connection in sampled_connections:
        start_name = connection["start"]
        end_name = connection["end"]

        start_rotations_arr = []
        end_rotations_arr = []
        for i in range(24):
            start_rotations_arr.append(storage.get_datapoint_data_selected_rotation_tensor_by_name(start_name, i))
            end_rotations_arr.append(storage.get_datapoint_data_selected_rotation_tensor_by_name(end_name, i))

        start_embeddings = seen_network.encoder_inference(torch.stack(start_rotations_arr).to(get_device()))
        end_embeddings = seen_network.encoder_inference(torch.stack(end_rotations_arr).to(get_device()))

        same_position_difference += torch.norm(start_embeddings[0] - start_embeddings[12], p=2, dim=0).mean().item()

        raw_diff = torch.norm(start_embeddings[0] - end_embeddings[12], p=2, dim=0).item()
        different_position_difference += raw_diff

        # if raw_diff > 1:
        #     different_position_difference -= raw_diff

    same_position_difference /= SAMPLES
    different_position_difference /= SAMPLES

    print(f"Same position difference: {same_position_difference}")
    print(f"Different position difference: {different_position_difference}")


def _get_connection_distances_seen_network(storage: StorageSuperset2, seen_network: any) -> any:
    connections = storage.get_all_connections_data()
    SAMPLES = min(1000, len(connections))
    seen_network.eval()
    seen_network = seen_network.to(get_device())

    sampled_connections = np.random.choice(np.array(connections), SAMPLES, replace=False)
    connections_distances_data = []

    start_data_arr = []
    end_data_arr = []

    for connection in sampled_connections:
        start_name = connection["start"]
        end_name = connection["end"]

        start_data = storage.get_datapoint_data_selected_rotation_tensor_by_name(start_name, 0)
        end_data = storage.get_datapoint_data_selected_rotation_tensor_by_name(end_name, 0)

        # start_data = storage.get_datapoint_data_random_rotation_tensor_by_name(start_name)
        # end_data = storage.get_datapoint_data_random_rotation_tensor_by_name(end_name)

        start_data_arr.append(start_data)
        end_data_arr.append(end_data)

    start_data_arr = torch.stack(start_data_arr).to(get_device())
    end_data_arr = torch.stack(end_data_arr).to(get_device())

    start_embedding = seen_network.encoder_inference(start_data_arr)
    end_embedding = seen_network.encoder_inference(end_data_arr)

    distance_data = torch.norm(start_data_arr - end_data_arr, p=2, dim=1)
    distance_embeddings = torch.norm(start_embedding - end_embedding, p=2, dim=1)

    length = len(distance_embeddings)

    for i in range(length):
        start_name = sampled_connections[i]["start"]
        end_name = sampled_connections[i]["end"]
        distance_real = sampled_connections[i]["distance"]
        distance_data_i = distance_data[i].item()
        distance_embeddings_i = distance_embeddings[i].item()
        connections_distances_data.append({
            "start": start_name,
            "end": end_name,
            "distance_real": distance_real,
            "distance_data": distance_data_i,
            "distance_embeddings": distance_embeddings_i
        })

    return connections_distances_data


def _get_connection_distances_adjacency_network_on_unknown_dataset(storage: StorageSuperset2,
                                                                   adjacency_network: any) -> any:
    adjacency_network.eval()
    adjacency_network = adjacency_network.to(get_device())

    datapoints = storage.get_all_datapoints()

    distances_arr = []
    start_data_arr = []
    end_data_arr = []

    set_pretty_display(len(datapoints), "Calculating distances")
    pretty_display_start()

    lng = len(datapoints)
    start_names = []
    end_names = []

    for i in range(lng):
        pretty_display(i)
        for j in range(i + 1, lng):
            start_name = datapoints[i]
            end_name = datapoints[j]
            distance = storage.get_datapoints_real_distance(start_name, end_name)

            start_data = storage.get_datapoint_data_selected_rotation_tensor_by_name(start_name, 0)
            end_data = storage.get_datapoint_data_selected_rotation_tensor_by_name(end_name, 0)
            # start_data = storage.get_datapoint_data_random_rotation_tensor_by_name(start_name)
            # end_data = storage.get_datapoint_data_random_rotation_tensor_by_name(end_name)

            start_names.append(start_name)
            end_names.append(end_name)
            start_data_arr.append(start_data)
            end_data_arr.append(end_data)
            distances_arr.append(distance)

    print("")
    print("finished first loop")
    start_data_arr = torch.stack(start_data_arr).to(get_device())
    end_data_arr = torch.stack(end_data_arr).to(get_device())
    distance_data = torch.norm(start_data_arr - end_data_arr, p=2, dim=1)
    adjacency_probabilities = adjacency_network(start_data_arr, end_data_arr)
    print("finished forwarding")

    set_pretty_display(len(adjacency_probabilities), "Calculating distances from thetas")
    pretty_display_start()

    predicted_adjacencies = []
    for idx, distance in enumerate(adjacency_probabilities):
        if distance[0] > 0.98:
            predicted_adjacencies.append(0)
        else:
            predicted_adjacencies.append(1)

    print("")
    index = 0

    good_predictions = 0
    bad_predictions = 0
    bad_pred_avg_distance = 0

    expected_good_predictions = 0

    neigh_net = load_manually_saved_ai("neigh_network_north.pth")
    neigh_net.eval()
    neigh_net = neigh_net.to(get_device())

    predicted_distances = []
    expected_distances = []

    for i in range(lng):
        for j in range(i + 1, lng):
            start_name = datapoints[i]
            end_name = datapoints[j]
            # print(start_names[index], end_names[index])
            distance_real = storage.get_datapoints_real_distance(start_name, end_name)

            predicted_adjacency = predicted_adjacencies[index]
            # 0 true, 1 false
            if predicted_adjacency == 0:
                if distance_real < 0.5:
                    good_predictions += 1


                elif distance_real > 1.25:
                    bad_predictions += 1
                    bad_pred_avg_distance += distance_real

                start_data = storage.get_datapoint_data_selected_rotation_tensor_by_name(start_name, 0).to(
                    get_device())
                end_data = storage.get_datapoint_data_selected_rotation_tensor_by_name(end_name, 0).to(get_device())
                distance_thetas = neigh_net(start_data.unsqueeze(0), end_data.unsqueeze(0)).squeeze(0)
                distance_percent = distance_thetas_to_distance_percent(distance_thetas)
                distance_percent *= MAX_DISTANCE

                predicted_distances.append(distance_percent)
                expected_distances.append(distance_real)
                if distance_real > 0.75:
                    print(f"Predicted distance: {distance_percent}, real distance: {distance_real}")

            if distance_real < 0.5:
                expected_good_predictions += 1

            index += 1

    se = 0
    for i in range(len(expected_distances)):
        se += (expected_distances[i] - predicted_distances[i]) ** 2

    se /= len(expected_distances)
    se = se ** 0.5
    print(f"Standard error: {se}")

    if bad_predictions > 0:
        bad_pred_avg_distance /= bad_predictions
        print(f"Bad predictions avg distance: {bad_pred_avg_distance}")

    print(f"Expected good predictions: {expected_good_predictions}")
    print(f"Good predictions: {good_predictions}")
    print(f"Bad predictions: {bad_predictions}")

    print("finished second loop")


def standard_error(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("Arrays must have the same length")

    # Compute squared differences
    squared_diff = np.square(np.array(y_true) - np.array(y_pred))

    # Calculate the standard error
    se = np.sqrt(np.mean(squared_diff))

    return se


def _get_connection_distances_neigh_network_on_unknown_dataset(storage: StorageSuperset2,
                                                               neighborhood_network: any) -> any:
    neighborhood_network.eval()
    neighborhood_network = neighborhood_network.to(get_device())

    datapoints = storage.get_all_datapoints()

    distances_arr = []
    start_data_arr = []
    end_data_arr = []

    set_pretty_display(len(datapoints), "Calculating distances")
    pretty_display_start()

    lng = len(datapoints)
    for i in range(lng):
        pretty_display(i)
        for j in range(i + 1, lng):
            start_name = datapoints[i]
            end_name = datapoints[j]
            distance = storage.get_datapoints_real_distance(start_name, end_name)
            if distance > 1:
                continue

            start_data = storage.get_datapoint_data_selected_rotation_tensor_by_name(start_name, 0)
            end_data = storage.get_datapoint_data_selected_rotation_tensor_by_name(end_name, 0)
            # start_data = storage.get_datapoint_data_random_rotation_tensor_by_name(start_name)
            # end_data = storage.get_datapoint_data_random_rotation_tensor_by_name(end_name)

            start_data_arr.append(start_data)
            end_data_arr.append(end_data)
            distances_arr.append(distance)

    print("")
    print("finished first loop")
    start_data_arr = torch.stack(start_data_arr).to(get_device())
    end_data_arr = torch.stack(end_data_arr).to(get_device())
    distance_data = torch.norm(start_data_arr - end_data_arr, p=2, dim=1)
    distances_thetas = neighborhood_network(start_data_arr, end_data_arr)
    print("finished forwarding")

    set_pretty_display(len(distances_thetas), "Calculating distances from thetas")
    pretty_display_start()

    predicted_distances = []
    print(len(distances_thetas))
    for idx, distance in enumerate(distances_thetas):
        pretty_display(idx)
        distance_percent = distance_thetas_to_distance_percent(distance)
        distance_percent *= MAX_DISTANCE
        predicted_distances.append(distance_percent)

    print("")
    print("finished making distajcces")

    final_connections = []
    cnt = 0
    for i in range(lng):
        for j in range(i + 1, lng):
            start_name = datapoints[i]
            end_name = datapoints[j]
            distance_real = storage.get_datapoints_real_distance(start_name, end_name)

            if distance_real > 1:
                # cnt += 1
                continue

            distance_data_i = distance_data[cnt].item()
            predicted_distance = predicted_distances[cnt].item()
            # predicted_distance = 0

            final_connections.append({
                "start": start_name,
                "end": end_name,
                "distance_real": distance_real,
                "distance_data": distance_data_i,
                "distance_embeddings": predicted_distance
            })
            cnt += 1

    print("finished second loop")

    return final_connections


def _get_connection_distances_neigh_network_on_training_dataset(storage: StorageSuperset2,
                                                                neighborhood_network: any) -> any:
    connections = storage.get_all_connections_data()
    SAMPLES = min(1000, len(connections))
    neighborhood_network.eval()
    neighborhood_network = neighborhood_network.to(get_device())

    sampled_connections = np.random.choice(np.array(connections), SAMPLES, replace=False)
    connections_distances_data = []

    start_data_arr = []
    end_data_arr = []

    cnt_bigger1 = 0
    cnt_smaller05 = 0

    for connection in sampled_connections:
        start_name = connection["start"]
        end_name = connection["end"]

        distance = storage.get_datapoints_real_distance(start_name, end_name)
        if distance > 1:
            cnt_bigger1 += 1
            continue
        elif distance < 0.5:
            cnt_smaller05 += 1
            continue

        start_data = storage.get_datapoint_data_selected_rotation_tensor_by_name(start_name, 0)
        end_data = storage.get_datapoint_data_selected_rotation_tensor_by_name(end_name, 0)

        # start_data = storage.get_datapoint_data_random_rotation_tensor_by_name(start_name)
        # end_data = storage.get_datapoint_data_random_rotation_tensor_by_name(end_name)

        start_data_arr.append(start_data)
        end_data_arr.append(end_data)

    print(f"Connections bigger than 1: {cnt_bigger1}")
    print(f"Connections smaller than 0.5: {cnt_smaller05}")

    start_data_arr = torch.stack(start_data_arr).to(get_device())
    end_data_arr = torch.stack(end_data_arr).to(get_device())
    distance_data = torch.norm(start_data_arr - end_data_arr, p=2, dim=1)
    distances_thetas = neighborhood_network(start_data_arr, end_data_arr)

    predicted_distances = []
    for distance in distances_thetas:
        distance_percent = distance_thetas_to_distance_percent(distance)
        distance_percent *= MAX_DISTANCE
        predicted_distances.append(distance_percent)

    for i in range(SAMPLES):
        start_name = sampled_connections[i]["start"]
        end_name = sampled_connections[i]["end"]
        distance_real = sampled_connections[i]["distance"]

        distance_data_i = distance_data[i].item()
        predicted_distance = predicted_distances[i].item()
        # predicted_distance = 0

        connections_distances_data.append({
            "start": start_name,
            "end": end_name,
            "distance_real": distance_real,
            "distance_data": distance_data_i,
            "distance_embeddings": predicted_distance
        })

    return connections_distances_data


def calculate_pearson_correlations(data):
    # Convert data to numpy array
    data_array = np.array(data)

    # Extract columns
    input_values = data_array[:, 0]
    second_number = data_array[:, 1]
    third_number = data_array[:, 2]

    # Pearson correlation
    pearson_input_second = pearsonr(input_values, second_number)
    pearson_input_third = pearsonr(input_values, third_number)
    pearson_second_third = pearsonr(second_number, third_number)

    # Print correlation results
    print("Pearson Correlation Coefficients:")
    print(f"Input vs Second Number: {pearson_input_second[0]:.3f} (p-value: {pearson_input_second[1]:.3e})")
    print(f"Input vs Third Number: {pearson_input_third[0]:.3f} (p-value: {pearson_input_third[1]:.3e})")
    print(f"Second Number vs Third Number: {pearson_second_third[0]:.3f} (p-value: {pearson_second_third[1]:.3e})")


if __name__ == "__main__":
    random_walk_datapoints = read_other_data_from_file(f"datapoints_random_walks_250_24rot.json")
    random_walk_connections = read_other_data_from_file(f"datapoints_connections_randon_walks_250_24rot.json")
    storage: StorageSuperset2 = StorageSuperset2()
    storage.incorporate_new_data(random_walk_datapoints, random_walk_connections)
    # neighborhood_distance_network = load_manually_saved_ai("adjacency_network_north.pth")
    neighborhood_distance_network = load_manually_saved_ai("adjacency_network_north_contrasted.pth")

    _get_connection_distances_adjacency_network_on_unknown_dataset(storage, neighborhood_distance_network)
    # connections = _get_connection_distances_neigh_network_on_training_dataset(storage, neighborhood_distance_network)
    # connections = _get_connection_distances_neigh_network_on_unknown_dataset(storage, neighborhood_distance_network)

    # Extract the data we need
    # filtered_connections = [
    #     connection for connection in connections
    # ]
    #
    # data_arr = [[connection["distance_real"], connection["distance_data"], connection["distance_embeddings"]] for
    #             connection in filtered_connections]
    #
    # # Convert to numpy array for easier manipulation
    # data_array = np.array(data_arr)
    #
    #
    # # Define a function to calculate relative difference
    # def relative_difference(a, b):
    #     return abs(a - b) / ((a + b) / 2)
    #
    #
    # # Set a threshold for what we consider a "big discrepancy"
    # threshold = 0.1
    #
    # filtered_data = data_array
    #
    # # Create the plot
    # plt.figure(figsize=(12, 4))
    #
    # # Plot Input vs Second Number
    # plt.subplot(131)
    # plt.scatter(filtered_data[:, 0], filtered_data[:, 1], color='blue', alpha=0.7)
    # plt.title('Input vs Second Number')
    # plt.xlabel('Input')
    # plt.ylabel('Second Number')
    #
    # # Plot Input vs Third Number
    # plt.subplot(132)
    # plt.scatter(filtered_data[:, 0], filtered_data[:, 2], color='red', alpha=0.7)
    # plt.title('Input vs Third Number')
    # plt.xlabel('Input')
    # plt.ylabel('Third Number')
    #
    # plt.tight_layout()
    # plt.show()
    #
    # # Print the number of points remaining after filtering
    # print(f"Before filtering: {len(data)}")
    # print(f"Number of points after filtering: {len(filtered_data)}")
    #
    # calculate_pearson_correlations(data_arr)
