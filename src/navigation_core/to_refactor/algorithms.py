from typing import Dict, List
from typing import Dict
import heapq
from src.navigation_core.pure_functions import connection_reverse_order
from src.runtime_storages.types import ConnectionAuthenticData, ConnectionSyntheticData


def _ensure_node_exists(node: str, connections_hashmap: Dict):
    if node not in connections_hashmap:
        connections_hashmap[node] = []


def build_connections_hashmap(connections: List[ConnectionAuthenticData | ConnectionSyntheticData],
                              exclude_datapoints: List[str]):
    connections_hashmap: Dict[str, List[ConnectionAuthenticData | ConnectionSyntheticData]] = {}
    connections_hashmap_names: Dict[str, List[str]] = {}

    for connection in connections:
        start = connection["start"]
        end = connection["end"]
        distance = connection["distance"]

        if start in exclude_datapoints or end in exclude_datapoints:
            continue

        _ensure_node_exists(start, connections_hashmap)
        _ensure_node_exists(end, connections_hashmap)
        _ensure_node_exists(start, connections_hashmap_names)
        _ensure_node_exists(end, connections_hashmap_names)

        if end not in connections_hashmap_names[start]:
            connections_hashmap[start].append(connection)
            connections_hashmap_names[start].append(end)

        if start not in connections_hashmap_names[end]:
            connections_hashmap[end].append(connection_reverse_order(connection))
            connections_hashmap_names[end].append(start)

    return connections_hashmap


def find_minimum_distance_between_datapoints_on_graph_djakstra(starting_point: str, ending_point: str,
                                                               connections_hashmap: Dict):
    """
    Finds the minimum distance between two datapoints on the graph with weighted edges
    Uses Dijkstra's algorithm
    """
    pq = [(0, starting_point)]  # Priority queue: (distance, node)
    distances = {starting_point: 0}
    visited = set()

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        if current_node == ending_point:
            return current_distance

        if current_node in visited:
            continue

        visited.add(current_node)

        for connection in connections_hashmap.get(current_node, []):
            neighbor = connection["end"]
            weight = connection["distance"]
            distance = current_distance + weight

            if distance < distances.get(neighbor, float("inf")):
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return float("inf")  # If ending_point is not reachable


def find_minimum_distance_between_datapoints_on_graph_bfs(starting_point: str, ending_point: str,
                                                          connections_hashmap: Dict):
    """
    Finds the minimum distance between two datapoints on the graph
    Uses BFS
    """
    queue = [(starting_point, 0, 0)]
    min_distances = {}
    min_distances[starting_point] = 0

    while len(queue) > 0:
        current, distance, edges = queue.pop(0)

        for connection in connections_hashmap[current]:
            end = connection["end"]
            if distance + connection["distance"] < min_distances.get(end, float("inf")):
                queue.append((connection["end"], distance + connection["distance"], edges + 1))
                min_distances[end] = distance + connection["distance"]

    return min_distances.get(ending_point, float("inf"))


def floyd_warshall_algorithm(connections_hashmap: Dict):
    """
    Floyd-Warshall algorithm for finding all pairs shortest path
    """
    nodes = list(connections_hashmap.keys())
    distances = {node: {node: float("inf") for node in nodes} for node in nodes}

    for node in nodes:
        distances[node][node] = 0

    for node in nodes:
        for connection in connections_hashmap[node]:
            distances[node][connection["end"]] = connection["distance"]

    # pretty_display_set_and_start(len(nodes))
    for idx, k in enumerate(nodes):
        # pretty_display(idx)
        for i in nodes:
            for j in nodes:
                distances[i][j] = min(distances[i][j], distances[i][k] + distances[k][j])

    return distances
