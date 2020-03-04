from MobilityVector import MobilityVector

class Taxi(object):
	__schedule_list = []
	request_list = []
	# 元素是Request对象
	path_node_list = []
	# 元素是(timestamp, lon, lat) 依timestamp递增顺序排序
	def __init__(self, taxi_id, cur_lon, cur_lat, init_last_update_time, partition_id_belongto, mobility_vector=None):
		self.taxi_id = taxi_id
		self.cur_lon = cur_lon
		self.cur_lat = cur_lat
		self.__schedule_list = []
		self.__last_update_time = init_last_update_time
		self.partition_id_belongto = partition_id_belongto
		self.mobility_vector = mobility_vector
	
	def update_status(self, moment):
		# 根据moment, __last_update_time和__schedule_list来更新taxi状态
		# 状态包括, taxi当前位置(经纬度), mobility vector
		# __last_update_time是用来更新状态的,
		self.__last_update_time = moment
		index = -1
		behind_all_timestamp = True
		for idx, path_node in enumerate(self.path_node_list):
			index = idx
			if moment < path_node[0]:
				behind_all_timestamp = False
				break
		
		if index == -1:# index==-1说明taxi无线路呆在原位，不用变化
			return
		
		
		if index == len(self.path_node_list) - 1 and behind_all_timestamp:   # index等于len-1的时候说明
			(self.cur_lon,self.cur_lat) = (self.path_node_list[index][1], self.path_node_list[index][2])
			self.path_node_list.clear()
			self.__schedule_list.clear()
			self.request_list.clear()
			self.mobility_vector = None
			
			# 还有partition

		(self.cur_lon,self.cur_lat) = (self.path_node_list[index-1][1], self.path_node_list[index-1][2])
		average_lon = average_lat = 0
		for req in self.request_list:
			average_lat += req.end_lat
			average_lon += req.end_lon
		tmp_len = len(self.request_list)
		average_lat /= tmp_len
		average_lon /= tmp_len
		self.mobility_vector = MobilityVector(self.cur_lon, self.cur_lat, average_lon, average_lat,"TAXI", self.taxi_id)