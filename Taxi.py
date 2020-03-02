class Taxi(object):
	__schedule_list = []
	def __init__(self, taxi_id, init_lon, init_lat):
		self.taxi_id = taxi_id
		self.init_lon = init_lon
		self.init_lat = init_lat
		self.__schedule_list = []