import math
import osmnx as ox
import pandas as pd
import pickle
import networkx as nx
map_file = open('./data/map.pickle', 'rb')
osm_map = pickle.load(map_file) # osm地图, 在判断距离某个经纬点最近的道路节点时可以使用

map_file.close()
tool_node_list = []
df = pd.read_csv('./data/node_list_with_cluster.csv')
tli = df.loc[:, 'real_id']
cluster_li = df.loc[:, 'cluster_id']
tmpp = [i for i in range(len(tli))]
tool_node_list = zip(tli, tmpp)
id_hash_map = dict(tool_node_list)

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