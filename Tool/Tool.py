import math
import osmnx as ox
import pandas as pd
import pickle
import networkx as nx
import time
map_file = open('./data/map.pickle', 'rb')
osm_map = pickle.load(map_file)  # osm地图, 在判断距离某个经纬点最近的道路节点时可以使用

map_file.close()
tool_node_list = []
df = pd.read_csv('./data/node_list_with_cluster.csv')
tli = df.loc[:, 'real_id']
cluster_li = df.loc[:, 'cluster_id']
tmpp = [i for i in range(len(tli))]
tool_node_list = zip(tli, tmpp)
id_hash_map = dict(tool_node_list)


SYSTEM_INIT_TIME = time.time()
EARLIEST_TIME = 1477929720   # 预先从数据库中查询出最早时间
TIME_OFFSET = SYSTEM_INIT_TIME - EARLIEST_TIME


def rad(deg):
    return (deg / 180.0) * math.pi


def get_distance(lon1, lat1, lon2, lat2):
    EARTH_RADIUS = 6378.137
    rad_lat1 = rad(lat1)
    rad_lat2 = rad(lat2)
    a = rad_lat1 - rad_lat2
    rad_lon1 = rad(lon1)
    rad_lon2 = rad(lon2)
    b = rad_lon2 - rad_lon1
    ret = 2 * math.asin(math.sqrt(math.pow(math.sin(a / 2), 2) +
                                  math.cos(rad_lat1) * math.cos(rad_lat2) * math.pow(math.sin(b / 2), 2)))
    ret *= EARTH_RADIUS
    ret = round(ret * 10000) / 10000
    return ret * 1000


def check_in_which_partition(lon, lat):
    ret = ox.get_nearest_node(osm_map, (lat, lon))
    ret = id_hash_map[ret]
    ret = cluster_li[ret]
    return ret


def get_shortest_path_node(node1, node2):
    return nx.shortest_path(osm_map, source=node1, target=node2, weight='length')


def get_shortest_path_length(node1, node2):
    return nx.shortest_path_length(osm_map, source=node1, target=node2, weight='length')


# 余弦相似度
def cosine_similarity(vec1, vec2):
    x = [vec1[2]-vec1[0], vec1[3]-vec1[1]]
    y = [vec2[2]-vec2[0], vec2[3]-vec2[1]]
    result1 = 0.0
    result2 = 0.0
    result3 = 0.0
    for i in range(len(x)):
        result1 += x[i]*y[i]  # sum(X*Y)
        result2 += x[i]**2  # sum(X*X)
        result3 += y[i]**2  # sum(Y*Y)

    return result1/((result2*result3)**0.5)

# def cosine_similarity(vec1, vec2):
#     x = [vec1[0], vec1[1], vec1[2], vec1[3]]
#     y = [vec2[0], vec2[1], vec2[2], vec2[3]]
#     sum_xy = 0.0
#     normX = 0.0
#     normY = 0.0
#     for a, b in zip(x, y):
#         sum_xy += a * b
#         normX += a ** 2
#         normY += b ** 2
#     if normX == 0.0 or normY == 0.0:
#         return None
#     else:
#         tmp = sum_xy / ((normX * normY) ** 0.5)
#         return tmp
