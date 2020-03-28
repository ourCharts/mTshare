import time
from Tool.Tool import *


TYPICAL_SPEED = 13.8889  # 单位是m/s


class Path:
    # 元素是(lon, lat) 依timestamp递增顺序排序
    def __init__(self,create_time):
        self.create_time = create_time
        # 此处的时间需要算偏移
        self.path_node_list = []


    def get_node_list(self):
        print('Showing node_list:')
        for node in self.path_node_list:
            print(node.lon,node.lat)


    def get_position(self, moment):
        index = self.is_over(moment)
        if index == -1:
            length = len(self.path_node_list)
            return self.path_node_list[length-1].lon, self.path_node_list[length-1].lat
        return index.lon, index.lat

        # -1是因为当前index是比moment大的，所以应返回前面一个才是比moment小的

    def is_over(self, moment):  # 结束返回-1，否则返回node对象
        delta_time = moment - self.create_time
        drive_distance = delta_time * TYPICAL_SPEED
        for idx, node in enumerate(self.path_node_list):
            length = get_shortest_path_length(self.path_node_list[0].node_id, node.node_id)
            if drive_distance > length:
                '''
                可能可以换成distance_matrix
                '''
                return self.path_node_list[idx-1]
        return -1
