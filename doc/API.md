### 变量集合
- node_list = []   Node对象
- taxi_list = []   Taxi对象
- node_distance_matrix：list，保存两个点之间的最短路距离
- landmark_list：list，保存的是三元元组，元组中的内容分别是经度、纬度、点在node_list中的id。第$i$个landmark对应的是第$i$个partition
- partition_list：list，保存的是Partition对象

### 对象

##### Node对象

属性：

* node_id：int，点的在osm地图上的id  
* lon：float，点的经度
* lat ：float，点的纬度
* cluster_id_belongto：int，当前点属于哪一个cluster/partition，存放的是cluster/partition的id

##### Taxi对象
属性：

* taxi_id：int，出租车的id，从0开始

* cur_lon：float，出租车当前的经度，会不断更新 
* cur_lat：float，出粗车当前的维度，会不断更新
* __last_update_time：float，私有成员，出租车上一次更新时的时间戳
* partition_id_belongto：int，出租车当前所属的partition的id
* seat_left：int，当前剩余多少座位。初始值为3。
* mobility_vector：出租车当前的mobility_vector
* schedule_list：list，出租车当前的行程规划，其中元素为行程规划点。行程规划点分三种，分别是起始点，终止点，初始点。起始点和终止点分别对应一个订单的起点和重点；初始点则是出租车最开始时没有订单所对应的行程规划点。一个行程规划点用一个字典实现，字典的内容为：
  * request_id：int，表示对应订单的id。若为初始点，则request_id为-1
  * schedule_type：str，表示行程规划点的类型。类型共三类：
    * DEPART
    * ARRIVAL
    * NO_ORDER
  * lon：float，行程规划点的经度
  * lat：float，行程规划点的纬度
  * arrival_time：float，行程规划点的预计到达时间的时间戳
* path：Path对象，出租车当前的路径
* capability：int，出租车的承载量。默认3

##### Partition对象
属性：

* partition_id：int，partition的id
* node_list：该partition内的点，存的是点在osm地图上的id
* taxi_list：该partition内的出租车，存的是出租车的id

##### Request对象
属性：

* request_id：int，订单的id，从0开始
* start_lon：float，起点的经度
* start_lat：float，起点的纬度
* end_lon：float，终点的经度
* end_lat：float，终点的纬度 
* start_node_id：int，距离起点最近的点在osm地图上的id
* end_node_id：int，距离终点最近的点在osm地图上的id
* release_time：float，订单的起始时间戳
* delivery_deadline：float，订单到达的ddl时间戳
* pickup_deadline：float，出租车去接订单的ddl时间戳

##### MobilityVector对象
属性：

* lon1：float，点1的经度
* lat1：float，点1的纬度
* lon2：float，点2的经度
* lat2 ：float，点2的纬度
* vector_type：mobility vector的类型
* ID：str，表示mobility vector的类型，如果vector_type是"TAXI", 那就是taxi的id; 如果vecotr_type是"REQ", 那就是request的id

##### Path对象
属性：

* create_time：float，路径的创建时间的时间戳

* path_node_list：list，存放路径上道路节点的真实id



##### main.py

