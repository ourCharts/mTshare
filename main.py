import pymysql
from Request import Request
import math
import time
import datetime
import pandas as pd
from Node import Node
from Taxi import Taxi
from MobilityVector import MobilityVector
import numpy as np
from Node import Node
from Partition import Partition
import osmnx as ox
import pickle
from Tool.Tool import *
from Path import Path

conn = pymysql.connect(host='127.0.0.1', user='root',
                       passwd='', db='taxidb', port=3308, charset='utf8')
cursor = conn.cursor(pymysql.cursors.SSCursor)

mobility_cluster = []
general_mobility_vector = []


TYPICAL_SPEED = 13.8889  # 单位是m/s
TAXI_TOTAL_NUM = 100
EARLIEST_TIME = 0   # 预先从数据库中查询出最早时间, 避免每次都要查询一次. 肯定不会是0, 最后会修改
TIME_OFFSET = 0
SYSTEM_INIT_TIME = 0

alpha = 0.999999921837146
node_list = []
taxi_list = []  # 里面包含有所有的taxi
taxi_status_queue = []  # taxi的事件队列
request_list = []
partition_list = []
landmark_list = []
node_distance = pd.read_csv('./data/node_distance.csv') # 该文件存放的是地图上所有道路节点点对之间的最短距离, 晚点放入
node_shortest_path = pd.read_csv('./data/node_shortest_path.csv') # 该文件存放的是地图上所有节点点对之间的最短路, 晚点放入 
node_distance_matrix = []


def cosine_similarity(x, y):
    sum_xy = 0.0
    normX = 0.0
    normY = 0.0
    for a, b in zip(x, y):
        sum_xy += a * b
        normX += a ** 2
        normY += b ** 2
    if normX == 0.0 or normY == 0.0:
        return None
    else:
        tmp = sum_xy / ((normX * normY) ** 0.5)
        if tmp < 0:
            return -tmp
        return tmp


def get_an_order(idx):
    sql = 'SELECT * FROM myorder ORDER BY start_time LIMIT %d, 1' % idx
    cursor.execute(sql)
    ret = cursor.fetchall()
    return ret


def system_init():
    taxi_table = pd.read_csv('./data/taxi_info_list.csv') # 晚点添加这个文件, 里面是taxi的基本信息, 包括每辆taxi的编号和初始经纬度
    # 初始化node_list, node_list中放的是Node对象
    df = pd.read_csv('./data/node_list_with_cluster.csv')
    for indexs in df.index:
        tmp = df.loc[indexs]
        node_list.append(
            Node(tmp['real_id'], tmp['lon'], tmp['lat'], tmp['cluster_id']))

    # 初始化landmark_list
    landmark_table = pd.read_csv('./data/landmark.csv') # 晚点添加...里面包含的内容是每个partition的landmark的经纬度.其下标与partition_list的下标一一对应
    landmark_list = zip(landmark_table.loc[:, 'lon'], landmark_table.loc[:, 'lat'])
    
    partition_list = [None] * (max(df.loc[:, 'cluster_id']) + 1)
    # 初始化所有partition实例
    for node_it in node_list:
        cid = node_it.cluster_id_belongto
        if partition_list[cid] is None:
            partition_list[cid] = Partition(cid, node_list=[], taxi_list=[])
            partition_list[cid].node_list.append(node_it.node_id)
        else:
            partition_list[cid].node_list.append(node_it.node_id)
    
    for taxi_it in taxi_table.index:
        tmp = taxi_table.loc[taxi_it]
        taxi_in_which_partition = check_in_which_partition(tmp['cur_lon'], tmp['cur_lat'])
        taxi_list.append(
            Taxi(tmp['taxi_id'], tmp['cur_lon'], tmp['cur_lat'], SYSTEM_INIT_TIME - TIME_OFFSET, partition_id_belongto=taxi_in_which_partition))
        partition_list[taxi_in_which_partition].taxi_list.append(tmp['taxi_id'])

    # 初始化邻接矩阵
    node_num = len(node_list)
    node_distance_matrix = [None] * node_num
    for i in range(node_num):
        node_distance_matrix[i] = [None] * len(node_num) 
    for i in range(node_num):
        for j in range(node_num):
            node_distance_matrix[i][j] = node_distance.iloc[i, j]


def request_fetcher(time_slot_start, time_slot_end):
    sql = 'SELECT * FROM myorder WHERE start_time >= {} AND start_time <= {}'.format(
        time_slot_start, time_slot_end)
    cursor.execute(sql)
    ret = cursor.fetchall()
    return ret


def update(request):
    for taxi_it in taxi_list:
        taxi_it.update_status(request['start_time'])
    mobility_cluster.clear()
    general_mobility_vector.clear()
    Lambda = 0.998
    for request_it in request_list:
        vec1 = [request_it.start_lon, request_it.start_lat, request_it.end_lon, request_it.end_lat]
        max_cos = -2
        max_idx = -1
        flag = False
        for idx, gene_it in enumerate(general_mobility_vector):
            cos_val = cosine_similarity(gene_it, vec1)
            # 计算出最相似的那个general_mobility_vector
            if cos_val > max_cos:
                max_idx = idx
                max_cos = cos_val
        if max_cos >= Lambda:
            flag = True
        if flag:
            mobility_cluster[max_idx].append(MobilityVector(
                vec1[0], vec1[1], vec1[2], vec1[3], 'REQ', request_it.request_id))
            x = y = z = w = 0
            for it in mobility_cluster[max_idx]:
                x += it[0]
                y += it[1]
                z += it[2]
                w += it[3]
            leng = len(mobility_cluster[max_idx])
            general_mobility_vector[max_idx] = MobilityVector(
                x / leng, y / leng, z / leng, w / leng, 'REQ', request_it.request_id)
        else:
            mobility_cluster.append([MobilityVector(
                vec1[0], vec1[1], vec1[2], vec1[3], 'REQ', request_it.request_id)])
            general_mobility_vector.append(MobilityVector(
                vec1[0], vec1[1], vec1[2], vec1[3], 'REQ', request_it.request_id))

    for taxi_it in taxi_list:
        vec2 = taxi_it.mobility_vector
        max_cos = -2
        max_idx = -1
        flag = False
        for idx, gene_it in enumerate(general_mobility_vector):
            cos_val = cosine_similarity(gene_it, vec2)
            if cos_val > max_cos:
                max_cos = cos_val
                max_idx = idx
        if max_cos >= Lambda:
            flag = True
        if flag:
            mobility_cluster[max_idx].append(MobilityVector(
                vec2[0], vec2[1], vec2[2], vec2[3], 'TAXI', taxi_it.taxi_id))
            x = y = z = w = 0
            for it in mobility_cluster[max_idx]:
                x += it[0]
                y += it[1]
                z += it[2]
                w += it[3]
            leng = len(mobility_cluster[max_idx])
            general_mobility_vector[max_idx] = MobilityVector(
                x / leng, y / leng, z / leng, w / leng, 'TAXI', taxi_it.taxi_id)
        else:
            mobility_cluster.append(
                [MobilityVector(vec2[0], vec2[1], vec2[2], vec2[3], 'TAXI', taxi_it.taxi_id)])
            general_mobility_vector.append(MobilityVector(
                vec2[0], vec2[1], vec2[2], vec2[3], 'TAXI', taxi_it.taxi_id))
    
    # 重置partition
    for par_it in partition_list:
        par_it.taxi_list.clear()
    for taxi_it in taxi_list:
        par_it[taxi_it.partition_id_belongto].append(taxi_it.taxi_id)


def taxi_req_matching(req: Request):
    u_lon, u_lat = req.start_lon, req.start_lat
    v_lon, v_lat = req.end_lon, req.end_lat
    nearest_start_id = ox.get_nearest_node(osm_map, (u_lat, u_lon))
    nearest_end_id = ox.get_nearest_node(osm_map, (v_lat, v_lon))
    delta_t = req.wait_time - node_distance_matrix[id_hash_map[nearest_start_id]][id_hash_map[nearest_end_id]] / TYPICAL_SPEED - req.release_time
    # 得到搜索范围的半径
    search_range = delta_t * TYPICAL_SPEED

    partition_intersected = set()
    for node_it in node_list:
        if node_it.cluster_id_belongto in partition_intersected:
            continue
        dis = get_distance(u_lon, u_lat, node_it.lon, node_it.lat)
        if dis <= search_range:
            partition_intersected.add(node_it.cluster_id_belongto)

    # 计算出PzLt
    taxi_in_intersected = []
    for it in partition_intersected:
        for taxi_it in partition_list[it].taxi_list: # partion对象中的taxi_list放的是taxi的id
            taxi_in_intersected.append(taxi_list[taxi_it].taxi_id) # 全局的taxi_list中放的是taxi对象, 故taxi_list[taxi_it].taxi_id是taxi的id

    vec = [req.start_lon, req.start_lat, req.end_lon, req.end_lat]
    max_cos = -2
    max_idx = -1
    for idx, gene_v in enumerate(general_mobility_vector):
        cos_val = cosine_similarity(gene_v, vec)
        if cos_val > max_cos:
            max_cos = cos_val
            max_idx =  idx
    C = mobility_cluster[max_idx]
    # 计算出CaLt
    C_li = []
    for it in C:
        if it.vector_type == 'TAXI':
            C_li.append(it.ID)
    # 取交集, 计算出所有候选taxi的list
    candidate_taxi = set(partition_intersected).intersection(set(C_li))

    """
    TODO
    1. 完成所有的matching的剩余部分, 即从候选taxi列表中, 通过minimum detour cost, 选出最合适的taxi
        !!!记得考虑空taxi
    2. 检查今天写的代码是否有bug
    """

def taxi_scheduling(candidate_taxi_list):
    pass


def basic_routing(selected_taxi):
    # 对匹配到的taxi进行路径规划
    taxi_path = Path()
    return taxi_path #一个Path对象


def main():
    system_init()
    order_index = 0
    SYSTEM_INIT_TIME = time.time()
    TIME_OFFSET = SYSTEM_INIT_TIME - EARLIEST_TIME
    last_time = SYSTEM_INIT_TIME - TIME_OFFSET  # 初始化为开始时间
    while True:
        now_time = time.time() - TIME_OFFSET
        reqs = request_fetcher(last_time, now_time)
        last_time = now_time
        if len(reqs) == 0:
            continue
        else:
            for req_item in reqs:
                end_time = req_item['start_time'] + datetime.timedelta(minutes=15)
                req_item = Request(req_item['order_id'], req_item['start_longitude'], req_item['start_latitude'], req_item['end_lon'], req_item['end_lat'], req_item['start_time'], end_time)
                request_list.append(req_item)
                # 用当前moment来更新所有taxi, mobility_cluster和general_cluster
                update(req_item)
                # selected_taxi = taxi_req_matching(req_item)
                


main()

cursor.close()
conn.close()
