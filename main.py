import pymysql
from Request import Request
import math
import time
import datetime
import pandas as pd
from Node import Node
from Taxi import Taxi

conn = pymysql.connect(host='127.0.0.1', user='root',
                       passwd='', db='taxidb', port=3308, charset='utf8')
cursor = conn.cursor(pymysql.cursors.SSCursor)

taxi_mobility_vector = []
request_mobility_vector = []
mobility_cluster = []
general_mobility_vector = []

TYPICAL_SPEED = 13.8889 # 单位是m/s
TAXI_TOTAL_NUM = 100

alpha = 0.999999921837146
node_list = []
taxi_list = [] # 里面包含有所有的taxi
taxi_status_queue = [] # taxi的事件队列


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


def determine(order):
    flag = False
    min_cos = 2
    min_idx = 0
    for idx, item in enumerate(general_mobility_vector):
        vec1 = (lon1, lat1, lon2, lat2) = order[0:4]
        cos_val = cosine_similarity(vec1, item)
        if cos_val >= alpha and cos_val <= min_cos:
            flag = True
            min_cos = cos_val
            min_idx = idx
    if not flag:
        # 新建一个mobility cluster
        mobility_cluster.append([order])
        general_mobility_vector.append(order)
    else:
        mobility_cluster[min_idx].append(order)
        a = b = c = d = 0
        leng = len(mobility_cluster[min_idx])
        for it in range(leng):
            a += mobility_cluster[min_idx][it][0]
            b += mobility_cluster[min_idx][it][1]
            c += mobility_cluster[min_idx][it][2]
            d += mobility_cluster[min_idx][it][3]
        general_mobility_vector[min_idx] = (a / leng, b / leng, c / leng, d / leng)    


def system_init():
    taxi_table = pd.read_csv('./data/taxi_info_list.csv') # 晚点添加这个文件, 里面是taxi的基本信息, 包括每辆taxi的编号和初始经纬度
    df = pd.read_csv('./data/node_list_with_cluster.csv')
    for indexs in df.index:
        tmp = df.loc[indexs]
        node_list.append(Node(tmp['real_id'], tmp['lon'], tmp['lat'], tmp['cluster_id']))
    for taxi_it in taxi_table.index:
        tmp = df.loc[taxi_it]
        taxi_list.append(Taxi(tmp['taxi_id'], tmp['init_lon'], tmp['init_lat']))



def main():
    system_init()
    order_index = 0
    while True:
        release_time = time.time()
        order_tuple = get_an_order(order_index)[0]
        order_id = order_tuple[0]
        order = order_tuple[3:8]
        order_index += 1
        determine(order)
        deadline_time = release_time + datetime.timedelta(minutes=10)
        req = Request(order_id, order[0], order[1], order[2], order[3], release_time, None, deadline_time)
        waiting_time = deadline_time - release_time
        search_range = waiting_time * TYPICAL_SPEED # 单位是m
        
        
        time.sleep(30)

main()

cursor.close()
conn.close()
