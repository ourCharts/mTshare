class Path:
    # 元素是(lon, lat) 依timestamp递增顺序排序    def __init__(self):
    path_node_list = []

    def get_position(self, moment):
        index = self.is_over(moment)
        if self.is_over(moment) < 0:
            length = len(self.path_node_list)
            return (self.path_node_list[length-1][1], self.path_node_list[length-1][2])
        return (self.path_node_list[index-1][1], self.path_node_list[index-1][2])
        # -1是因为当前index是比moment大的，所以应返回前面一个才是比moment小的

    def is_over(self, moment):  # 结束返回-1，否则返回序号
        for idx, path_node in enumerate(self.path_node_list):
            if moment < path_node[0]:
                return idx
        return -1
