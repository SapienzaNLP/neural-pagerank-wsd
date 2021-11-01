from tqdm.auto import tqdm
from statistics import mean
import random
import json
import os

import networkx as nx

graph_path = os.path.join(os.getcwd(), 'data', 'wn_graph.json')
with open(graph_path) as f:
    graph_dict = json.load(f)
print(f'[INFO] Graph is loaded from "{graph_path}".')

synsets_indices = graph_dict['indices']
synsets_lst, related_synsets_lst = synsets_indices[0], synsets_indices[1]

graph_ = nx.Graph()
for i in range(len(synsets_lst)):
    graph_.add_edge(synsets_lst[i], related_synsets_lst[i])
print('[INFO] Graph is built.')


n_samples = 1000
with open('graph_info_output.txt', encoding='utf-8', mode='w+') as f:
    for component in tqdm(nx.connected_components(graph_), desc='looping graphs'):
        component_ = graph_.subgraph(component)
        nodes = component_.nodes()
        lengths = []
        for _ in tqdm(range(n_samples), desc='sampling', leave=False):
            n1, n2 = random.choices(list(nodes), k=2)
            length = nx.shortest_path_length(component_, source=n1, target=n2)
            lengths.append(length)
        mean_shortest_path = mean(lengths)
        f.write(
            f'Nodes #: {len(nodes)}, mean of shortest path: {mean_shortest_path} \n')
        # # print(f'diameter={component_.diameter()}')


# G = graph_
# shortest_paths = []
# number_of_nodes = []
# diameters = []
# connected_subgraphs = [G.subgraph(cc) for cc in nx.connected_components(G)]
# print('[INFO] connected subgraphs fetched.')
# for _graph in connected_subgraphs:
#     n_nodes = _graph.number_of_nodes()
#     number_of_nodes.append(n_nodes)
#     print(f'Number of nodes: {n_nodes}')

#     avg_shortest_path = average_shortest_path_length(_graph)
#     shortest_paths.append(avg_shortest_path)
#     print(f'Avg Shortest path len: {avg_shortest_path}')

#     dia = diameter(_graph)
#     diameters.append(dia)
#     print(f'Diameter: {dia}')

#     print('____________________________' * 3)


# with open('shortest_paths.pkl', 'wb') as f_shortest_paths:
#     pickle.dump(shortest_paths, f_shortest_paths)

# with open('number_of_nodes.pkl', 'wb') as f_number_of_nodes:
#     pickle.dump(number_of_nodes, f_number_of_nodes)

# with open('diameters.pkl', 'wb') as f_diameters:
#     pickle.dump(diameters, f_diameters)

# TAKES A LOT OF TIME
# G = graph_
# connected_subgraphs = [G.subgraph(cc) for cc in nx.connected_components(G)]
# print('[INFO] connected subgraphs fetched.')
# Gcc = max(nx.connected_components(G), key=len)
# giantC = G.subgraph(Gcc)
# print('[INFO] Fetched Giant Subgraph')

# MORE TIME EFFICIENT TO RETRIEVE THE GRAPH THAN PREVIOUS BLOCK
# G = graph_
# connected_subgraphs = [G.subgraph(cc) for cc in nx.connected_components(G)]
# print('[INFO] connected subgraphs fetched.')
# Gcc = max(nx.connected_components(G), key=len)
# giantC = G.subgraph(Gcc)
# print('[INFO] Fetched Giant Subgraph')

# print(f'Avg Shortest path len: {average_shortest_path_length(giantC)}')
# print(f'Diameter: {diameter(giantC)}')
