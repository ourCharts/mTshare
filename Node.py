class Node:
    def __init__(self, node_id, lon, lat, cluster_id_belongto: int):
        self.node_id = node_id
        self.lon = lon
        self.lat = lat
        self.cluster_id_belongto = cluster_id_belongto