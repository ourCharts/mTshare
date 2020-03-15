# import osmnx as ox
# import pickle
# import networkx as nx
# import pandas as pd

# import matplotlib.pyplot as plt

# df = pd.read_csv('./data/node_list_with_cluster.csv')
# tli = df.loc[:, 'real_id']
# map_file = open('./data/map.pickle', 'rb')
# print('1111')
# osm_map = pickle.load(map_file)
# path = nx.bidirectional_shortest_path(osm_map,source=tli[0],target=tli[10])
# G = nx.Graph()


# for idx,item in enumerate(path):
#     G.add_node(item)
#     if idx!=0:
#         G.add_edge(path[idx],path[idx-1])
# print(path)
# print('2222')
# nx.draw(G,node_size = 1,edge_color='#555555')
# nx.draw(osm_map,node_size = 0.7,edge_color='#ffffff')
# print('asd')
# plt.show()
arr = [1,4,2,4,5]
arr = set(arr)
arr1 = [4,2]
arr1 = set(arr1)
result = arr.difference(arr1)
print(result)