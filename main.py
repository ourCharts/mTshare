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
import os
import glob
import copy
from tqdm import tqdm



def get_an_order(idx):
    sql = 'SELECT * FROM myorder ORDER BY start_time LIMIT %d, 1' % idx
    cursor.execute(sql)
    ret = cursor.fetchall()
    return ret


def system_init():
    print('System Initiating...')
    taxi_table = pd.read_csv('./data/taxi_info_list.csv')
    df = pd.read_csv('./data/node_list_with_cluster.csv')
    for indexs in df.index:
        tmp = df.loc[indexs]
        node_list.append(
            Node(tmp['real_id'], tmp['lon'], tmp['lat'], int(tmp['cluster_id'])))

    # .里面包含的内容是每个partition的landmark的经纬度.其下标与partition_list的下标一一对应
    landmark_table = pd.read_csv('./data/landmark.csv')
    global landmark_list
    landmark_list = list(
        zip(landmark_table.loc[:, 'lon'], landmark_table.loc[:, 'lat']))

    global partition_list
    partition_list = [None] * (max(df.loc[:, 'cluster_id']) + 1)
    # 初始化所有partition实例
    for node_it in node_list:
        cid = node_it.cluster_id_belongto
        if partition_list[cid] is None:
            partition_list[cid] = Partition(cid, node_list=[], taxi_list=[])
            partition_list[cid].node_list.append(int(node_it.node_id))
        else:
            partition_list[cid].node_list.append(int(node_it.node_id))

    global taxi_list
    for taxi_it in taxi_table.index:
        tmp = taxi_table.loc[taxi_it]
        taxi_in_which_partition = check_in_which_partition(
            tmp['init_lon'], tmp['init_lat'])
        taxi_list.append(
            Taxi(int(tmp['taxi_id']), tmp['init_lon'], tmp['init_lat'], SYSTEM_INIT_TIME - TIME_OFFSET, partition_id_belongto=taxi_in_which_partition, seat_left=3))
        partition_list[taxi_in_which_partition].taxi_list.append(
            int(tmp['taxi_id']))

    # 初始化邻接矩阵
    global node_distance_matrix
    node_distance_matrix = copy.copy(node_distance.values)
    # print(node_distance_matrix)
    print('done')


def request_fetcher(time_slot_start, time_slot_end):
    sql = 'SELECT * FROM myorder WHERE start_time >= {} AND start_time <= {}'.format(
        time_slot_start, time_slot_end)
    cursor.execute(sql)
    ret = cursor.fetchall()
    return ret


def update(request):
    print('In update')
    for taxi_it in taxi_list:
        # print('Updating each taxi\'s status')
        taxi_it.update_status(request.release_time)
    mobility_cluster.clear()
    general_mobility_vector.clear()
    for request_it in request_list:
        vec1 = [request_it.start_lon, request_it.start_lat,
                request_it.end_lon, request_it.end_lat]
        max_cos = -2
        max_idx = -1
        flag = False
        for idx, gene_it in enumerate(general_mobility_vector):
            cos_val = cosine_similarity(
                [gene_it.lon1, gene_it.lat1, gene_it.lon2, gene_it.lat2], vec1)
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
                x += it.lon1
                y += it.lat1
                z += it.lon2
                w += it.lat2
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
        if len(vec2) == 0 or taxi_it.seat_left == taxi_it.capability:
            continue
        max_cos = -2
        max_idx = -1
        flag = False
        for idx, gene_it in enumerate(general_mobility_vector):
            cos_val = cosine_similarity(
                [gene_it.lon1, gene_it.lat1, gene_it.lon2, gene_it.lat2], vec2)
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
                x += it.lon1
                y += it.lat1
                z += it.lon2
                w += it.lat2
            leng = len(mobility_cluster[max_idx])
            general_mobility_vector[max_idx] = MobilityVector(
                x / leng, y / leng, z / leng, w / leng, 'TAXI', taxi_it.taxi_id)
        else:
            mobility_cluster.append(
                [MobilityVector(vec2[0], vec2[1], vec2[2], vec2[3], 'TAXI', taxi_it.taxi_id)])
            general_mobility_vector.append(MobilityVector(
                vec2[0], vec2[1], vec2[2], vec2[3], 'TAXI', taxi_it.taxi_id))

    # 重置partition
    global partition_list
    # print(partition_list)
    for par_it in partition_list:
        par_it.taxi_list.clear()
    for taxi_it in taxi_list:
        # print(taxi_it.partition_id_belongto)
        partition_list[taxi_it.partition_id_belongto].taxi_list.append(
            int(taxi_it.taxi_id))


def taxi_req_matching(req: Request):
    print('In taxi req matching')
    u_lon, u_lat = req.start_lon, req.start_lat
    v_lon, v_lat = req.end_lon, req.end_lat
    req_start_node = ox.get_nearest_node(osm_map, (u_lat, u_lon))
    nearest_start_id = ox.get_nearest_node(osm_map, (u_lat, u_lon))
    nearest_end_id = ox.get_nearest_node(osm_map, (v_lat, v_lon))
    delta_t = req.delivery_deadline - node_distance_matrix[id_hash_map[nearest_start_id]
                                                           ][id_hash_map[nearest_end_id]] / TYPICAL_SPEED - req.release_time
    # 得到搜索范围的半径
    search_range = delta_t * TYPICAL_SPEED
    print('search range is {}'.format(search_range))
    partition_intersected = set()
    for node_it in node_list:
        if node_it.cluster_id_belongto in partition_intersected:
            continue
        dis = get_shortest_path_length(req_start_node, node_it.node_id)
        if dis <= search_range:
            partition_intersected.add(node_it.cluster_id_belongto)

    # 计算出PzLt
    taxi_in_intersected = []
    for it in partition_intersected:
        # print('one of the intersected partition: {}'.format(it))
        # partion对象中的taxi_list放的是taxi的id
        for taxi_it in partition_list[it].taxi_list:
            # print('taxi_it = {}'.format(taxi_it))
            """
            bug: partition_intersected中的taxi都是一样的
            """
            if taxi_list[taxi_it].is_available:
                # 全局的taxi_list中放的是taxi对象, 故taxi_list[taxi_it].taxi_id是taxi的id
                taxi_in_intersected.append(taxi_list[taxi_it].taxi_id)

    if len(taxi_in_intersected) == 0:  # 在规定时间内没有taxi能来，所以放弃订单
        return None                    # 放弃订单了
    vec = [req.start_lon, req.start_lat, req.end_lon, req.end_lat]
    max_cos = -2
    max_idx = -1
    for idx, gene_v in enumerate(general_mobility_vector):
        cos_val = cosine_similarity(
            [gene_v.lon1, gene_v.lat1, gene_v.lon2, gene_v.lat2], vec)
        if cos_val > max_cos:
            max_cos = cos_val
            max_idx = idx
    # if max_idx == -1:  # 说明cluster为空，所以CaLt是空的
    #     candidate_taxi = taxi_in_intersected
    #     print('Empty CaLt!')
    '''
        cluster不可能为空
    '''
    # 计算出CaLt
    C = mobility_cluster[max_idx]
    C_li = []
    for it in C:
        if it.vector_type == 'TAXI':
            C_li.append(it.ID)
    best_candidate_taxi = set(taxi_in_intersected).intersection(
        set(C_li))  # 取交集, 计算出所有候选taxi的list
    secondary_candidate_taxi = set(taxi_in_intersected).difference(set(C_li))

    best_candidate_taxi = list(best_candidate_taxi)
    secondary_candidate_taxi = list(secondary_candidate_taxi)

    return best_candidate_taxi, secondary_candidate_taxi
    # best_candidate_taxi是mv也符合的车，secondary_candidate_taxi是能到达但mv不符合的车


def insertion_feasibility_check(taxi_id, req: Request, pos_i, pos_j):  # 在前面插入
    print('In insertion feasibility check')
    req_start_node_id = req.start_node_id
    req_end_node_id = req.end_node_id

    pre_node_lon = (taxi_list[taxi_id].schedule_list[pos_i - 1])['lon']
    pre_node_lat = (taxi_list[taxi_id].schedule_list[pos_i - 1])['lat']
    pre_node_id = ox.get_nearest_node(osm_map, (pre_node_lat, pre_node_lon))

    aft_node_lon = (taxi_list[taxi_id].schedule_list[pos_i])['lon']
    aft_node_lat = (taxi_list[taxi_id].schedule_list[pos_j])['lat']
    aft_node_id = ox.get_nearest_node(osm_map, (aft_node_lat, aft_node_lon))

    ddl = 0
    dis = node_distance_matrix[id_hash_map[pre_node_id]][id_hash_map[req_start_node_id]
                                                         ] + node_distance_matrix[id_hash_map[req_start_node_id]][id_hash_map[aft_node_id]]
    req_id = (taxi_list[taxi_id].schedule_list[pos_i])['request_id']
    if (taxi_list[taxi_id].schedule_list[pos_i])['schedule_type'] == 'ARRIVAL':
        ddl = request_list[req_id].delivery_deadline
    elif (taxi_list[taxi_id].schedule_list[pos_i])['schedule_type'] == 'DEPART':
        ddl = request_list[req_id].pickup_deadline

    if (taxi_list[taxi_id].schedule_list[pos_i - 1])['arrival_time'] + node_distance_matrix[id_hash_map[pre_node_id]][id_hash_map[req_start_node_id]] / TYPICAL_SPEED > ddl:
        return False

    for i in range(pos_i, len(taxi_list[taxi_id].schedule_list)):
        req_id = (taxi_list[taxi_id].schedule_list[i])['request_id']
        ddl_ = 0
        if (taxi_list[taxi_id].schedule_list[i])['schedule_type'] == 'ARRIVAL':
            ddl_ = request_list[req_id].delivery_deadline
        elif (taxi_list[taxi_id].schedule_list[i])['schedule_type'] == 'DEPART':
            ddl_ = request_list[req_id].pickup_deadline

        tmp = (taxi_list[taxi_id].schedule_list[i])[
            'arrival_time'] + dis / TYPICAL_SPEED
        if tmp > ddl_:
            return False
        (taxi_list[taxi_id].schedule_list[i])['arrival'] += tmp

    pre_node_lon = (taxi_list[taxi_id].schedule_list[pos_j - 1])['lon']
    pre_node_lat = (taxi_list[taxi_id].schedule_list[pos_j - 1])['lat']
    pre_node_id = ox.get_nearest_node(osm_map, (pre_node_lat, pre_node_lon))

    aft_node_lon = (taxi_list[taxi_id].schedule_list[pos_j])['lon']
    aft_node_lat = (taxi_list[taxi_id].schedule_list[pos_j])['lat']
    aft_node_id = ox.get_nearest_node(osm_map, (aft_node_lat, aft_node_lon))

    ddl = 0
    dis = node_distance_matrix[id_hash_map[pre_node_id]][id_hash_map[req_end_node_id]
                                                         ] + node_distance_matrix[id_hash_map[req_end_node_id]][id_hash_map[aft_node_id]]
    req_id = (taxi_list[taxi_id].schedule_list[pos_j])['request_id']
    if (taxi_list[taxi_id].schedule_list[pos_j])['schedule_type'] == 'ARRIVAL':
        ddl = request_list[req_id].delivery_deadline
    elif (taxi_list[taxi_id].schedule_list[pos_j])['schedule_type'] == 'DEPART':
        ddl = request_list[req_id].pickup_deadline

    if (taxi_list[taxi_id].schedule_list[pos_j - 1])['arrival_time'] + node_distance_matrix[id_hash_map[pre_node_id]][id_hash_map[req_end_node_id]] / TYPICAL_SPEED > ddl:
        return False

    for i in range(pos_j, len(taxi_list[taxi_id].schedule_list)):
        req_id = (taxi_list[taxi_id].schedule_list[i])['request_id']
        ddl_ = 0
        if (taxi_list[taxi_id].schedule_list[i])['schedule_type'] == 'ARRIVAL':
            ddl_ = request_list[req_id].delivery_deadline
        elif (taxi_list[taxi_id].schedule_list[i])['schedule_type'] == 'DEPART':
            ddl_ = request_list[req_id].pickup_deadline

        tmp = (taxi_list[taxi_id].schedule_list[i])[
            'arrival'] + dis / TYPICAL_SPEED
        if tmp > ddl_:
            return False
        (taxi_list[taxi_id].schedule_list[i])['arrival'] += tmp

    return True


def partition_filter(node1, node2):  # 返回一个数组，组成元素是partition id
    print('In partition filter')
    # 根据论文P7
    partition1 = check_in_which_partition(node1['lon'], node1['lat'])
    partition2 = check_in_which_partition(node2['lon'], node2['lat'])
    if partition1 == partition2:
        return [partition_list[partition1]]
    # print('landmark_list is {}'.format(landmark_list))
    # print('partition1 is {}'.format(partition1))
    landmark1 = landmark_list[partition1]
    # print('partition2 is {}'.format(partition2))
    landmark2 = landmark_list[partition2]

    node1 = ox.get_nearest_node(osm_map, (landmark1[0], landmark1[1]))
    node2 = ox.get_nearest_node(osm_map, (landmark2[0], landmark2[1]))

    # lm1到lm2的travel cost
    cost_1to2 = node_distance_matrix[id_hash_map[node1]
                                     ][id_hash_map[node2]] / TYPICAL_SPEED
    forever_mobility_vector = [landmark1[0],
                               landmark1[1], landmark2[0], landmark2[1]]

    filtered_partition = []
    for idx, one_partition in enumerate(partition_list):
        if partition1 == idx:  #如果指向起点的partition就跳过
            continue
        tmp_lm = landmark_list[idx]
        tmp_vec = [landmark1[0], landmark1[1], tmp_lm[0], tmp_lm[1]]
        print('我在332-------------------------')
        print(partition1,idx)
        debug1 = partition_list[partition1]
        debug2 = partition_list[idx]
        if debug1 == debug2:
            print(True)
        else:
            print(False)
        debug1 = landmark_list[partition1]
        debug2 = landmark_list[idx]
        if debug1 == debug2:
            print(True)
        else:
            print(False)            
        print(tmp_vec)
        # Travel direction rule 来自论文P7左栏
        if cosine_similarity(tmp_vec, forever_mobility_vector) < Lambda:
            continue
        # Travel cost rule
        tmp_node = ox.get_nearest_node(osm_map, (tmp_lm[0], tmp_lm[1]))
        cost_1totmp = node_distance_matrix[id_hash_map[node1]][id_hash_map[tmp_node]] / TYPICAL_SPEED
        cost_tmpto2 = node_distance_matrix[id_hash_map[tmp_node]][id_hash_map[node2]] / TYPICAL_SPEED
        if cost_1totmp + cost_tmpto2 <= (1 + partition_filter_param) * cost_1to2:
            filtered_partition.append((one_partition, cost_1totmp))

    filtered_partition.sort(key=lambda x: x[1])
    # 使filtered_partition里的元组根据cost_1totmp递增的顺序排列

    filtered_partition = [i[0] for i in filtered_partition]
    '''
        注意第一个partition是不是taxi所在的地方
        最后一个partition是不是终点的地方
        感觉要将第一个和最后一个地方换成确切的lon lat而不是partition
    
    '''
    return filtered_partition


def basic_routing(Slist, taxi_it):
    print('In basic routing')
    # 根据论文P7
    taxi_path = Path()
    sum_path_distance = 0
    for idx, s_node in enumerate(Slist):
        if idx == len(Slist) - 1:
            break
        path_distance = 0
        filtered_partition = partition_filter(Slist[idx], Slist[idx-1])

        for index, p_node in enumerate(filtered_partition):
            if index == len(filtered_partition) - 1:
                break

            # 得到partition id在partition_list中的下表
            node1 = partition_list.index(filtered_partition[index])
            node1_landmark = landmark_list[node1]
            # 得到partition id在partition_list中的下表
            node2 = partition_list.index(filtered_partition[index+1])
            node2_landmark = landmark_list[node2]

            length = len(taxi_path.path_node_list)
            print(node1_landmark,node2_landmark)
            print('-------------------')
            tmp_list = get_shortest_path_node(ox.get_nearest_node(osm_map,(node1_landmark[0],node1_landmark[1])), 
                        ox.get_nearest_node(osm_map,(node2_landmark[0],node2_landmark[1])))

            tmp_list = [Node(x, node_list[id_hash_map[x]].lon, node_list[id_hash_map[x]].lat,
                             node_list[id_hash_map[x]].cluster_id_belongto) for x in tmp_list]
            taxi_path.path_node_list[length: ] = tmp_list
            """
            landmark上的点就是osm地图上的节点, 所以可以直接调用get_shortest_path_length
            """
            path_distance += get_shortest_path_length(node1_landmark, node2_landmark)

        Slist[idx]['arrival_time'] = now_time + path_distance / TYPICAL_SPEED

        sum_path_distance += path_distance
        # 获得两个partition的landmark的最短路径
    print('395*****************')
    print(taxi_list[taxi_it].cur_lon)
    print(taxi_list[taxi_it].cur_lat)
    print('395*****************')
    taxi_pos_node = ox.get_nearest_node(
        osm_map, (taxi_list[taxi_it].cur_lon, taxi_list[taxi_it].cur_lat))
    taxi_to_first_slist_node_path = get_shortest_path_node(
        taxi_pos_node, taxi_path.path_node_list[0].node_id)
    taxi_path.path_node_list.insert(0, taxi_to_first_slist_node_path)
    # 加上了taxi目前位置到slist第一个节点的路径,因为上面的路径是不包括taxi原本位置的，只包括了slist里面的

    sum_path_distance += get_shortest_path_length(
        taxi_pos_node, taxi_path.path_node_list[0][0])
    # 加上了taxi目前位置到slist第一个节点的路径长度,因为上面的路径是不包括taxi原本位置的，只包括了slist里面的

    path_cost = sum_path_distance / TYPICAL_SPEED
    return (taxi_path, path_cost)  # 一个Path对象和Path的cost


def possibility_routing(Slist, taxi_it):
    return 1, 2


def taxi_scheduling(candidate_taxi_list, req, mode=1):
    print('In taxi scheduling')
    possible_insertion = []
    minimum_cost = 10 ** 10
    selected_taxi = -1
    selected_taxi_path = None
    res = []
    print("我在423***************")
    print(taxi_list[0].cur_lat)
    print(len(candidate_taxi_list))
    print(candidate_taxi_list)
    print("我在423***************")
    for taxi_it in candidate_taxi_list:
        print('In taxi scheduling iterating candidate')
        possible_insertion.clear()
        bnd = len(taxi_list[taxi_it].schedule_list)
        if bnd == 1:
            possible_insertion.append((1, 2))
        else:
            for i in range(1, bnd):
                for j in range(i + 1, bnd):
                    flag = insertion_feasibility_check(taxi_it, req, i, j)
                    if flag:
                        possible_insertion.append((i, j))

        ori_cost = taxi_list[taxi_it].cur_total_cost
        for insertion in possible_insertion:
            Slist = copy.deepcopy(taxi_list[taxi_it].schedule_list)

            start_point = {'request_id': req.request_id, 'schedule_type': 'DEPART', 'lon': req.start_lon,
                           'lat': req.start_lat, 'arrival_time': None}  # arrival_time在之后routing的时候确定
            end_point = {'request_id': req.request_id, 'schedule_type': 'ARRIVAL',
                         'lon': req.end_lon, 'lat': req.end_lat, 'arrival_time': None}
            print('type of Slist is {}'.format(type(Slist)))
            Slist.insert(insertion[0], start_point)
            Slist.insert(insertion[1], end_point)

            if mode:
                print('mode is basic routing')
                new_path, cost = basic_routing(
                    Slist, taxi_it)  # 写完basic routing就ok了
            else:
                new_path, cost = possibility_routing(Slist, taxi_it)
            if cost - ori_cost < minimum_cost:
                res = Slist
                minimum_cost = cost - ori_cost
                selected_taxi = taxi_it
                selected_taxi_path = new_path

    taxi_list[selected_taxi].schedule_list = copy.deepcopy(res)
    if not selected_taxi_path:
        taxi_list[selected_taxi].path.path_node_list = []
        return selected_taxi, None
    else:
        taxi_list[selected_taxi].path.path_node_list = selected_taxi_path
        return selected_taxi, selected_taxi_path


# ==================================全局变量==============================================


conn = pymysql.connect(host='127.0.0.1', user='root',
                       passwd='', db='tenman', port=3308, charset='utf8')
cursor = conn.cursor(pymysql.cursors.SSCursor)

mobility_cluster = []
general_mobility_vector = []


TYPICAL_SPEED = 13.8889  # 单位是m/s
TAXI_TOTAL_NUM = 100
REQUESTS_TO_PROCESS = 100  # 总共要处理多少个request
partition_filter_param = 1.0

Lambda = 0.9
# alpha = 0.999999921837146
node_list = []  # Node对象
taxi_list = []  # Taxi对象
taxi_status_queue = []  # taxi的事件队列
request_list = []
partition_list = []
landmark_list = []

files = glob.glob('./data/node_distance/node_distance_*.csv')
node_distance = pd.read_csv(files[0])
node_distance = node_distance.loc[:, ~
                                  node_distance.columns.str.contains('^Unnamed')]
for idx in range(1, len(files)):
    tmp = pd.read_csv(files[idx])
    tmp = tmp.drop(['Unnamed: 0'], axis=1)
    node_distance = node_distance.append(tmp)

node_distance_matrix = []
now_time = 0
# ==================================全局变量==============================================


def main():
    req_cnt = 0
    system_init()
    order_index = 0
    print(landmark_list[2])
    print(landmark_list[11])
    last_time = SYSTEM_INIT_TIME - TIME_OFFSET  # 初始化为开始时间
    while True:
        input('我在518行，按下回车开始')
        print(taxi_list[0].cur_lon,taxi_list[0].cur_lat)
        if req_cnt > REQUESTS_TO_PROCESS:
            break
        now_time = time.time() - TIME_OFFSET
        reqs = request_fetcher(last_time, now_time)
        last_time = now_time
        if len(reqs) == 0:
            continue
        else:
            for req_item in tqdm(reqs, desc='Processing requests...'):
                end_time = req_item[1] + \
                    datetime.timedelta(minutes=15).seconds
                """
                0: order_id,
                1: start_time,
                2: end_time,
                3: start_longitude,
                4: start_latitude,
                5: end_longitude,
                6: end_latitude
                """
                start_node_id = ox.get_nearest_node(
                    osm_map, (req_item[4], req_item[3]))

                end_node_id = ox.get_nearest_node(
                    osm_map, (req_item[4], req_item[3]))
                print(id_hash_map[start_node_id])
                print(id_hash_map[end_node_id])
                time_on_tour = node_distance_matrix[id_hash_map[start_node_id]
                                                    ][id_hash_map[end_node_id]]

                req_item = Request(req_cnt, req_item[3], req_item[4], req_item[5],
                                   req_item[6], start_node_id, end_node_id, req_item[1], req_item[2])  # 打车的时候难道还能给你输入到达的ddl的吗???????????
                req_cnt += 1
                req_item.config_pickup_deadline(
                    req_item.delivery_deadline - time_on_tour)
                request_list.append(req_item)
                # 用当前moment来更新所有taxi, mobility_cluster和general_cluster
                update(req_item)
                candidate_taxi_list, secondary_candidate_list = taxi_req_matching(
                    req_item)
                print('candidate: ')
                print(candidate_taxi_list)
                print('secondary: ')
                print(secondary_candidate_list)
                # 如果没有候选taxi会返回none
                # print(candidate_taxi_list is None)
                if len(candidate_taxi_list) != 0:
                    taxi_scheduling(candidate_taxi_list, req_item, 1)
                else:
                    taxi_scheduling(secondary_candidate_list, req_item, 1)


main()

cursor.close()
conn.close()
