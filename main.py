import pymysql
from Request import Request
import math
import time
import datetime
import pandas as pd
from Node import Node
from Taxi import Taxi
import Queue
from MobilityVector import MobilityVector
import numpy as np

conn = pymysql.connect(host='127.0.0.1', user='root',
                       passwd='', db='taxidb', port=3308, charset='utf8')
cursor = conn.cursor(pymysql.cursors.SSCursor)

request_mobility_vector = []
mobility_cluster = []
general_mobility_vector = []

TYPICAL_SPEED = 13.8889  # 单位是m/s
TAXI_TOTAL_NUM = 100
EARLIEST_TIME = 0   # 预先从数据库中查询出最早时间, 避免每次都要查询一次. 肯定不会是0, 最后会修改
TIME_OFFSET = 0
SYSTEM_INIT_TIME = 0
ORDER_COUNT = 0

alpha = 0.999999921837146
node_list = []
taxi_list = []  # 里面包含有所有的taxi
taxi_status_queue = []  # taxi的事件队列
mobility_cluster = []
request_list = []


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
    ORDER_COUNT = 0
    # 晚点添加这个文件, 里面是taxi的基本信息, 包括每辆taxi的编号和初始经纬度
    taxi_table = pd.read_csv('./data/taxi_info_list.csv')
    df = pd.read_csv('./data/node_list_with_cluster.csv')
    for indexs in df.index:
        tmp = df.loc[indexs]
        node_list.append(
            Node(tmp['real_id'], tmp['lon'], tmp['lat'], tmp['cluster_id']))
    for taxi_it in taxi_table.index:
        tmp = df.loc[taxi_it]
        taxi_list.append(
            Taxi(tmp['taxi_id'], tmp['init_lon'], tmp['init_lat'], SYSTEM_INIT_TIME - TIME_OFFSET))


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
    for request_it in request_list:
        vec1 = request_it[3:7]
        min_cos = 2
        min_idx = -1
        for idx, gene_it in enumerate(general_mobility_vector):
            cos_val = cosine_similarity(gene_it, vec1)
            if cos_val < min_cos:
                min_idx = idx
                min_cos = cos_val
        if min_idx != -1:
            mobility_cluster[min_idx].append(MobilityVector(
                vec1[0], vec1[1], vec1[2], vec1[3], 'REQ', request_it.request_id))
            x = y = z = w = 0
            for it in mobility_cluster[min_idx]:
                x += it[0]
                y += it[1]
                z += it[2]
                w += it[3]
            leng = len(mobility_cluster[min_idx])
            general_mobility_vector[min_idx] = MobilityVector(
                x / leng, y / leng, z / leng, w / leng, 'REQ', request_it.request_id)
        else:
            mobility_cluster.append([MobilityVector(
                vec1[0], vec1[1], vec1[2], vec1[3], 'REQ', request_it.request_id)])
            general_mobility_vector.append(MobilityVector(
                vec1[0], vec1[1], vec1[2], vec1[3], 'REQ', request_it.request_id))

    for taxi_it in taxi_list:
        vec2 = taxi_it.mobility_vector
        min_cos = 2
        min_idx = -1
        for idx, gene_it in enumerate(general_mobility_vector):
            cos_val = cosine_similarity(gene_it, vec2)
            if cos_val < min_cos:
                min_cos = cos_val
                min_idx = idx
        if min_idx != -1:
            mobility_cluster[min_idx].append(MobilityVector(
                vec2[0], vec2[1], vec2[2], vec2[3], 'TAXI', taxi_it.taxi_id))
            x = y = z = w = 0
            for it in mobility_cluster[min_idx]:
                x += it[0]
                y += it[1]
                z += it[2]
                w += it[3]
            leng = len(mobility_cluster[min_idx])
            general_mobility_vector[min_idx] = MobilityVector(
                x / leng, y / leng, z / leng, w / leng, 'TAXI', taxi_it.taxi_id)
        else:
            mobility_cluster.append(
                [MobilityVector(vec2[0], vec2[1], vec2[2], vec2[3], 'TAXI', taxi_it.taxi_id)])
            general_mobility_vector.append(MobilityVector(
                vec2[0], vec2[1], vec2[2], vec2[3], 'TAXI', taxi_it.taxi_id))


def taxi_req_matching(req):
    # taxi和request匹配
    pass


def taxi_routing(selected_taxi):
    # 对匹配到的taxi进行路径规划
    pass


def main():
    system_init()
    order_index = 0
    SYSTEM_INIT_TIME = time.time()
    TIME_OFFSET = SYSTEM_INIT_TIME - EARLIEST_TIME
    last_time = SYSTEM_INIT_TIME - TIME_OFFSET  # 初始化为开始时间
    while True:
        now_time = time.time() - TIME_OFFSET
        last_time = now_time
        reqs = request_fetcher(last_time, now_time)
        if len(reqs) == 0:
            continue
        else:
            for req_item in reqs:
                request_list.append(req_item)
                # 用当前moment来更新所有taxi, mobility_cluster和general_cluster
                update(req_item)
                selected_taxi = taxi_req_matching(req)


main()

cursor.close()
conn.close()
