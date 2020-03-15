import time
from Tool.Tool import *


TYPICAL_SPEED = 13.8889  # 单位是m/s


class Path:
    # 元素是(lon, lat) 依timestamp递增顺序排序
    def __init__(self):
        self.create_time = time.time()
        # 此处的时间需要算偏移
        self.path_node_list = []

    def get_position(self, moment):
        index = self.is_over(moment)
        if self.is_over(moment) < 0:
            length = len(self.path_node_list)
            return self.path_node_list[length-1][1], self.path_node_list[length-1][2]
            '''这里是nodeid，要问lyl如何变成经纬度'''
        return index.lon, index.lat

        # -1是因为当前index是比moment大的，所以应返回前面一个才是比moment小的

    def is_over(self, moment):  # 结束返回-1，否则返回经纬度
        delta_time = moment - self.create_time
        drive_distance = delta_time * TYPICAL_SPEED
        for idx, node in enumerate(self.path_node_list):
            if idx == 0:
                continue
            if drive_distance > get_shortest_path_length(self.path_node_list[0], node):
                return self.path_node_list[idx-1]
        return -1
