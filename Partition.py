import logging

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.ERROR)
handle = logging.FileHandler('./log/partition_log.txt')
handle.setLevel(level=logging.ERROR)
fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handle.setFormatter(fmt)
logger.addHandler(handle)

class Partition(object):
    __node_list = []
    def __init__(self, partition_id, landmark_node_id, landmark_lon, landmark_lat, node_list, taxi_list):
        self.partition_id = partition_id
        self.landmark_node_id = landmark_node_id
        self.landmark_lon = landmark_lon
        self.landmark_lat = landmark_lat
        self.__node_list = node_list
        self.__taxi_list = taxi_list
    def update_taxi_list(self, taxi_id, op_type):
        if op_type  == 'APPEND':
            self.__taxi_list.append(taxi_id)
        elif op_type == 'REMOVE':
            try:
                self.__taxi_list.remove(taxi_id)
            except ValueError:
                logger.error('Can\'t remove the taxi. Maybe the taxi doesn\'t exist in the partition')