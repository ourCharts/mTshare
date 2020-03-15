from MobilityVector import MobilityVector
from Tool.Tool import *
from Path import Path


class Taxi:
    cur_total_cost = 0

    def __init__(self, taxi_id, cur_lon, cur_lat, init_last_update_time, partition_id_belongto, seat_left, mobility_vector=[]):
        self.seat_left = (3 if seat_left == None else seat_left)
        self.taxi_id = taxi_id
        self.cur_lon = cur_lon
        self.cur_lon = cur_lat
        self.schedule_list = [{'request_id': -1, 'schedule_type': 'NO_ORDER',
                               'lon': cur_lon, 'lat': cur_lat, 'arrival_time': init_last_update_time}]
        # schedule list中保存的是字典, 里面的内容包括: request_id: request_id, schedule_type: shedule的类型(出发或到达), lon: 经度, lat: 纬度, arrival_time: 计算出来的预期到达时间
        self.__last_update_time = init_last_update_time
        self.partition_id_belongto = partition_id_belongto
        self.mobility_vector = mobility_vector
        self.path = Path()  # 元素是(lon, lat)
        self.cur_total_cost = 0
        self.seat_left = seat_left
        self.capability = 3

    def is_available(self):
        if self.seat_left > 0:
            return True
        return False

    def update_schedule(self, moment):
        for idx, schedule_node in enumerate(self.schedule_list):
            if schedule_node['arrival_time'] < moment:
                del self.schedule_list[idx]

    def update_status(self, moment):
        # 状态： cur_lon、cur_lon、__last_update_time
        #		 schedule_list 、partition_id_belongto、mobility_vector

        self.__last_update_time = moment
        self.update_schedule(moment)
        # 更新经纬度
        print(self.path)
        if len(self.path.path_node_list) == 0:
            return
        self.cur_lon, self.cur_lat = self.path.get_position(moment)
        self.partition_id_belongto = check_in_which_partition(
            self.cur_lon, self.cur_lat)
        if self.path.is_over(moment) < 0:
            self.path = Path()
            self.schedule_list = [{'request_id': -1, 'schedule_type': 'NO_ORDER',
                                   'lon': self.cur_lon, 'lat': self.cur_lat, 'arrival_time': self.__last_update_time}]
            self.mobility_vector = []
            return

        # mobility-vector的更新
        average_lon = average_lat = 0
        sum_item = 0
        for sch_node in self.schedule_list:
            if sch_node['schedule_type'] != 'ARRIVAL':
                continue
            average_lat += sch_node['lon']
            average_lon += sch_node['lat']
            sum_item += 1
        average_lat /= sum_item
        average_lon /= sum_item
        self.mobility_vector = MobilityVector(
            self.cur_lon, self.cur_lat, average_lon, average_lat, "TAXI", self.taxi_id)
