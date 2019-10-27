# encoding: utf8

import os
import math
import json
import networkx as nx
import functools

g_subway_graph = None


def geo_distance(origin, destination):
    """
    Calculate the Haversine distance.

    Parameters
    ----------
    origin : tuple of float
        (lat, long)
    destination : tuple of float
        (lat, long)

    Returns
    -------
    distance_in_km : float

    Examples
    --------
    >>> origin = (48.1372, 11.5756)  # Munich
    >>> destination = (52.5186, 13.4083)  # Berlin
    >>> round(distance(origin, destination), 1)
    504.2
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d


def load_data(filepath):
    """
    load subway data
    """
    with open(filepath) as f:
        subway_data = f.read()

    subway_data = json.loads(subway_data)

    return subway_data


def cal_path_distance(graph, path):
    """
    calculate path distance.
    """
    distance = 0.0

    for i, station in enumerate(path):
        if i == 0:
            continue

        distance += graph.edges[path[i-1], path[i]]['distance']

    return distance


def stats_transfer_times_and_stations(graph, path):
    """
    Statistics transfer times and stations.
    """
    transfer_times = 0
    transfer_stations = []

    for i, station in enumerate(path):
        if i < 2:
            continue

        if graph.edges[path[i-1], path[i]]['line_name'] != \
                graph.edges[path[i-2], path[i-1]]['line_name']:
            transfer_times += 1
            transfer_stations.append(path[i-1])

    return transfer_times, transfer_stations


def debug_path(graph, path):
    """
    debug path info.
    """
    print('path {}'.format('->'.join(path)))
    print('distance {}'.format(cal_path_distance(graph, path)))
    print('station number {}'.format(len(path)))
    print('transfer times {}'.format(stats_transfer_times_and_stations(graph, path)[0]))


def optimal_search(graph, start, dest, search_strategy=lambda graph, pathes: pathes):
    """
    Optimal search with strategy.
    """
    # check whether the station existed?
    assert start in graph, '{} not found in graph, please check station name first!'.format(start)
    assert dest in graph, '{} not found in graph, please check station name first!'.format(dest)

    pathes = [[start]]
    visited = set()

    while pathes:
        path = pathes.pop(0)

        frointer = path[-1]
        # if frointer in visited:
        #     continue

        if frointer == dest:
            # print(pathes)
            return path

        visited.add(frointer)

        for successor in graph[frointer]:
            if successor in path:
                continue  # check loop

#             if successor == dest:  # may be the sub optimal path
#                 return path + [successor]

            new_path = path + [successor]  # enqueue
            pathes.append(new_path)

        # apply search strategy
        pathes = search_strategy(graph, pathes) if search_strategy else pathes

    print('Do not find the path between {} and {}'.format(start, dest))
    return None


def build_subway_graph(subway_data: dict, line_name=None):
    """
    Build subway graph.
    """
    subway_graph = nx.Graph()

    # add station nodes and edges
    for line in subway_data:
        for station in line['stations']:
            subway_graph.add_node(station['name'], **station)

    for line in subway_data:
        for i, station in enumerate(line['stations']):
            if i == 0:
                continue

            last = line['stations'][i-1]
            distance = geo_distance(last['geo'], station['geo'])
            subway_graph.add_edge(last['name'], station['name'],
                                  distance=distance, line_name=line['line_name'])

        # process loop line
        if line['is_loop'] == '1':
            begin = line['stations'][0]
            end = line['stations'][-1]
            distance = geo_distance(begin['geo'], end['geo'])
            subway_graph.add_edge(begin['name'], end['name'],
                                  distance=distance, line_name=line['line_name'])

    return subway_graph


def sort_by_cost(graph, pathes, c1=1, c2=0, c3=0):
    def cost(path):
        return c1 * cal_path_distance(graph, path) \
            + c2 * len(path[1:]) \
            + c3 * stats_transfer_times_and_stations(graph, path)[0]

    return sorted(pathes, key=cost)


def search(start, dest, strategey=None):
    return optimal_search(g_subway_graph, start, dest, strategey)


def init_graph():
    SUBWAY_DATA_FILEPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bj_subway.json')
    subway_data = load_data(SUBWAY_DATA_FILEPATH)
    subway_graph = build_subway_graph(subway_data)

    global g_subway_graph
    g_subway_graph = subway_graph


if __name__ == '__main__':
    START = '西二旗'
    DEST = '北京西站'
    init_graph()
    path = search(START, DEST, functools.partial(sort_by_cost, c1=1, c2=0, c3=4))
    debug_path(g_subway_graph, path)
