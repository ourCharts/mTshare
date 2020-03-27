from MobilityVector import MobilityVector
from Tool.Tool import *
from Path import Path


class Taxi:
    cur_total_cost = 0

    def __init__(self, taxi_id, cur_lon, cur_lat, init_last_update_time, partition_id_belongto, seat_left, mobility_vector=None):
        self.seat_left = (3 if seat_left == None else seat_left)
        self.taxi_id = taxi_id
        self.cur_lon = cur_lon
        self.cur_lat = cur_lat
        self.schedule_list = [{'request_id': -1, 'schedule_type': 'NO_ORDER',
                               'lon': cur_lon, 'lat': cur_lat, 'arrival_time': init_last_update_time}]
        # schedule list中保存的是字典, 里面的内容包括: request_id: request_id, schedule_type: shedule的类型(出发或到达), lon: 经度, lat: 纬度, arrival_time: 计算出来的预期到达时间
        self.__last_update_time = init_last_update_time
        self.partition_id_belongto = partition_id_belongto
        self.mobility_vector = mobility_vector
        self.path = Path()  
        self.cur_total_cost = 0
        self.seat_left = seat_left
        self.capability = self.seat_left


    def show_schedule(self):
        print('showing schedule: This is taxi {}'.format(self.taxi_id))
        for idx,node in enumerate(self.schedule_list):
            print('{}. {},经纬度：{},{}, 到达时间：{}'.format(idx,node['schedule_type'],node['lon'],node['lat'],node['arrival_time']))


    def show_path_list(self):
        print('showing schedule: This is taxi {}'.format(self.taxi_id))
        self.path.get_node_list()


    def is_available(self):
        if self.seat_left > 0:
            return True
        return False

    def is_empty(self):
        if self.seat_left == self.capability:
            return True
        return False

    def update_schedule(self, moment):
        
        if len(self.schedule_list) == 1 and self.schedule_list[0]['request_id'] == -1:
            return
        # self.show_schedule()
        del_list =[]
        for idx, schedule_node in enumerate(self.schedule_list):
            if schedule_node['arrival_time'] < moment:
                del_list.append(idx)
        for i in range(len(del_list)-1,-1,-1):
            del self.schedule_list[del_list[i]]

    def update_status(self, moment):
        # 状态： cur_lon、cur_lon、__last_update_time
        #		 schedule_list 、partition_id_belongto、mobility_vector
        self.__last_update_time = moment
              
        self.update_schedule(moment)
        # 更新经纬度
        if len(self.path.path_node_list) == 0:
            return
        print('taxi id is {}'.format(self.taxi_id))
        print('before updating, the position is lon: {}, lat: {}'.format(self.cur_lon, self.cur_lat))
        self.cur_lon, self.cur_lat = self.path.get_position(moment)
        print('after updating, the position is lon: {}, lat: {}'.format(self.cur_lon, self.cur_lat))
        self.partition_id_belongto = check_in_which_partition(
            self.cur_lon, self.cur_lat)
        print('这里是74行-------------------------------')
        self.show_schedule()
        print('这里是74行-------------------------------')
        if self.path.is_over(moment) == -1 or len(self.schedule_list) == 0:
            self.path = Path()
            self.schedule_list = [{'request_id': -1, 'schedule_type': 'NO_ORDER',
                                   'lon': self.cur_lon, 'lat': self.cur_lat, 'arrival_time': self.__last_update_time}]
            self.mobility_vector = None
            print('update over')
            return

        # mobility-vector的更新
        average_lon = average_lat = 0
        sum_item = 0
        print()
        self.show_schedule()
        for sch_node in self.schedule_list:
            print('DDDDDDDDDDDDDDDDDD')
            print('schedule_type is {}'.format(sch_node['schedule_type']))
            if sch_node['schedule_type'] != 'ARRIVAL':
                continue
            print('line 89!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            average_lat += sch_node['lon']
            average_lon += sch_node['lat']
            sum_item += 1
        average_lat /= sum_item
        average_lon /= sum_item
        self.mobility_vector = MobilityVector(
            self.cur_lon, self.cur_lat, average_lon, average_lat, "TAXI", self.taxi_id)
        print('update over')
