class Taxi(object):
	__schedule_list = []
	def __init__(self, taxi_id, cur_lon, cur_lat, init_last_update_time, partition_id_belongto, mobility_vector=None):
		self.taxi_id = taxi_id
		self.cur_lon = cur_lon
		self.cur_lat = cur_lat
		self.__schedule_list = []
		self.__last_update_time = init_last_update_time
		self.partition_id_belongto = partition_id_belongto
		self.mobility_vector = mobility_vector
	
	def update_status(self, moment):
		# 根据moment, __last_update_status和__schedule_list来更新taxi状态
		# 状态包括, taxi当前位置(经纬度), mobility vector
		# __last_update_time是用来更新状态的,
		pass