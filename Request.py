class Request:
    def __init__(self, request_id, start_lon, start_lat, end_lon, end_lat, release_time=None, delivery_deadline=None):
        self.request_id = request_id
        self.start_lon = start_lon
        self.start_lat = start_lat
        self.end_lon = end_lon
        self.end_lat = end_lat
        self.release_time = release_time
        self.delivery_deadline = delivery_deadline
