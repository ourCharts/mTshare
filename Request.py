class Request:
    pickup_deadline = 0

    def __init__(self, request_id, start_lon, start_lat, end_lon, end_lat, start_node_id, end_node_id, release_time=None, delivery_deadline=None, pickup_deadline=None):
        self.request_id = request_id
        self.start_lon = start_lon
        self.start_lat = start_lat
        self.end_lon = end_lon
        self.end_lat = end_lat
        self.start_node_id = start_node_id
        self.end_node_id = end_node_id
        self.release_time = release_time
        self.delivery_deadline = delivery_deadline

    def config_pickup_deadline(self, t):
        self.pickup_deadline = t
