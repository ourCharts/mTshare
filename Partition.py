import logging

# 这段打印日志的代码可以删掉, 感觉没什么用
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.ERROR)
handle = logging.FileHandler('./log/partition_log.txt')
handle.setLevel(level=logging.ERROR)
fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handle.setFormatter(fmt)
logger.addHandler(handle)

class Partition(object):
    def __init__(self, partition_id, node_list, taxi_list):
        self.partition_id = partition_id
        self.node_list = node_list
        self.taxi_list = taxi_list
    def update_taxi_list(self, taxi_id, op_type):
        if op_type  == 'APPEND':
            self.taxi_list.append(taxi_id)
        elif op_type == 'REMOVE':
            try:
                self.taxi_list.remove(taxi_id)
            except ValueError:
                logger.error('Can\'t remove the taxi. Maybe the taxi doesn\'t exist in the partition')