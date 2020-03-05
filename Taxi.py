from MobilityVector import MobilityVector
from Tool.Tool import *
from Path import Path
class Taxi:
	__schedule_list = []
	request_list = []# 元素是Request对象
	def __init__(self, taxi_id, cur_lon, cur_lat, init_last_update_time, partition_id_belongto,seat_left = 3, mobility_vector=None,path = None):
		self.taxi_id = taxi_id
		self.cur_lon = cur_lon
		self.cur_lon = cur_lat
		self.__schedule_list = []
		self.__last_update_time = init_last_update_time
		self.partition_id_belongto = partition_id_belongto
		self.mobility_vector = mobility_vector
		self.path = path# 元素是(timestamp, lon, lat) 依timestamp递增顺序排序
		self.seat_left = seat_left
	

	def is_available(self):
		if seat_left > 0:
			return True
		return False

	def update_schedule(self, moment):
		pass
	'''
		迟些要更新__schedule_list，要看看后面routing怎么写
	'''	

	def update_status(self, moment):
		# 状态： cur_lon、cur_lon、__last_update_time
		#		 __schedule_list 、partition_id_belongto、mobility_vector

		self.__last_update_time = moment
		self.update_schedule(moment)
		# 更新经纬度
		if not self.path:
			return
		self.cur_lon, self.cur_lat = self.path.get_position(moment)
		self.partition_id_belongto = check_in_which_partition(self.cur_lon,self.cur_lat)
		if self.path.is_over() < 0:
			self.path = None
			self.request_list.clear()
			self.mobility_vector = None
			return
		
		# mobility-vector的更新
		average_lon = average_lat = 0
		for req in self.request_list:
			average_lat += req.end_lat
			average_lon += req.end_lon
		tmp_len = len(self.request_list)
		average_lat /= tmp_len
		average_lon /= tmp_len
		self.mobility_vector = MobilityVector(self.cur_lon, self.cur_lat, average_lon, average_lat,"TAXI", self.taxi_id)