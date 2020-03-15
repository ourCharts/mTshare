class MobilityVector:
    def __init__(self, lon1, lat1, lon2, lat2, vector_type, ID):
        self.lon1 = lon1
        self.lat1 = lat1
        self.lon2 = lon2
        self.lat2 = lat2
        self.vector_type = vector_type
        self.ID = ID   # 如果vector_type是"TAXI", 那就是taxi的id; 如果vecotr_type是"REQ", 那就是request的id
