import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer, normalize
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import math


non_zero_system_call_list = []
non_zero_car_count_list = []
non_zero_body_count_list = []
non_zero_mariadb_counts_list = []
non_zero_mysql_counts_list = []
non_zero_postgres_counts_list = []
non_zero_sysbench_counts_list = []

name_list = []
distance_list = []
dict_of_names = {}

list1 = ['car', 'body', 'mariadb', 'mysql', 'postgres', 'mongo', 'redis', 'wordpress', 'caddy', 'httpd', 'sysbench']
list2 = ['car', 'body', 'mariadb', 'mysql', 'postgres', 'mongo', 'redis', 'wordpress', 'caddy', 'httpd', 'sysbench']

list_1d = []
list_2d = []

df = pd.read_csv ('compared_555.csv')

system_call_list     = df['system_call'].to_list()
car_counts_list       = df['car_count'].to_list()
body_counts_list      = df['body_count'].to_list()
mariadb_counts_list  = df['mariadb_counts'].to_list()
mysql_counts_list    = df['mysql_counts'].to_list()
postgres_counts_list = df['postgres_counts'].to_list()
mongo_counts_list = df['mongo_counts'].to_list()
redis_counts_list = df['redis_counts'].to_list()
wordpress_counts_list = df['wordpress_counts'].to_list()
caddy_counts_list = df['caddy_counts'].to_list()
httpd_counts_list = df['httpd_counts'].to_list()
sysbench_counts_list = df['sysbench_counts'].to_list()


sc = StandardScaler()
#sc = Normalizer()


# car_counts_list = normalize([car_counts_list], norm="max")
# body_counts_list = normalize([body_counts_list], norm="max")
# mariadb_counts_list = normalize([mariadb_counts_list], norm="max")
# mysql_counts_list = normalize([mysql_counts_list], norm="max")
# postgres_counts_list = normalize([postgres_counts_list], norm="max")
# sysbench_counts_list = normalize([sysbench_counts_list], norm="max")


# car_counts_list = normalize([car_counts_list], norm="l1")
# body_counts_list = normalize([body_counts_list], norm="l1")
# mariadb_counts_list = normalize([mariadb_counts_list], norm="l1")
# mysql_counts_list = normalize([mysql_counts_list], norm="l1")
# postgres_counts_list = normalize([postgres_counts_list], norm="l1")
# sysbench_counts_list = normalize([sysbench_counts_list], norm="l1")


car_counts_list = normalize([car_counts_list], norm="l2").tolist()[0]
#print(type(car_counts_list)).tolist()[0]

#print(car_counts_list.shape)
body_counts_list = normalize([body_counts_list], norm="l2").tolist()[0]
mariadb_counts_list = normalize([mariadb_counts_list], norm="l2").tolist()[0]
mysql_counts_list = normalize([mysql_counts_list], norm="l2").tolist()[0]
postgres_counts_list = normalize([postgres_counts_list], norm="l2").tolist()[0]
mongo_counts_list = normalize([mongo_counts_list], norm="l2").tolist()[0]
redis_counts_list = normalize([redis_counts_list], norm="l2").tolist()[0]
wordpress_counts_list = normalize([wordpress_counts_list], norm="l2").tolist()[0]
caddy_counts_list = normalize([caddy_counts_list], norm="l2").tolist()[0]
httpd_counts_list = normalize([httpd_counts_list], norm="l2").tolist()[0]
sysbench_counts_list = normalize([sysbench_counts_list], norm="l2").tolist()[0]
print(car_counts_list)
print(body_counts_list)


#print(body_counts_list)
#normalized_df['car'] = pd.Series(car_counts_list)

normalized_df = pd.DataFrame(list(zip(car_counts_list, body_counts_list, mariadb_counts_list, mysql_counts_list, postgres_counts_list,
                                      mongo_counts_list, redis_counts_list, wordpress_counts_list, caddy_counts_list, httpd_counts_list, sysbench_counts_list)))
#normalized_df.columns = ['car', 'body', 'mariadb', 'mysql', 'postgres', 'sysbench']
#normalized_df = normalized_df.T
#print(normalized_df)
#exit()
#print(normalized_df.take([5]))
#normalized_df = pd.concat([normalized_df, df_individual_train], axis=0)

# print("CPNMV:",np.array([car_counts_list]))
# print(np.array([car_counts_list]).shape)
# exit()

dict_of_names['car_counts_list'] = np.array([car_counts_list])
dict_of_names['body_counts_list'] = np.array([body_counts_list])
dict_of_names['mariadb_counts_list'] = np.array([mariadb_counts_list])
dict_of_names['mysql_counts_list'] = np.array([mysql_counts_list])
dict_of_names['postgres_counts_list'] = np.array([postgres_counts_list])
dict_of_names['mongo_counts_list'] = np.array([mongo_counts_list])
dict_of_names['redis_counts_list'] = np.array([redis_counts_list])
dict_of_names['wordpress_counts_list'] = np.array([wordpress_counts_list])
dict_of_names['caddy_counts_list'] = np.array([caddy_counts_list])
dict_of_names['httpd_counts_list'] = np.array([httpd_counts_list])
dict_of_names['sysbench_counts_list'] = np.array([sysbench_counts_list])
#car_count_list = np.reshape(car_count_list,(-1,1))
#body_count_list = np.reshape(body_count_list,(-1,1))
#print("HELLO:",dict_of_names['car_counts_list'])

#exit()


#car_count_list = sc.fit_transform(car_count_list)
#body_count_list = sc.fit_transform(body_count_list)

#print(car_count_list)


for iterator1 in list1:
    for iterator2 in list2:
        name = 'dist_' + iterator1 +'_' + iterator2
        candidate_1 = iterator1 + '_counts_list'
        candidate_2 = iterator2 + '_counts_list'
        #print(candidate_1)
        #print(candidate_2)
        dist = [np.linalg.norm(x-y) for x,y in zip(dict_of_names[candidate_1], dict_of_names[candidate_2])][0]
        dist = format(dist,".2f")
        #print(name, dist)
        name_list.append(name)
        distance_list.append(dist)
        list_1d.append(dist)
    list_2d.append(list_1d)
    list_1d = []

#print(list_2d)

print(list1)
for i in list_2d:
    print(*i)


exit()
# dist_car_body = [np.linalg.norm(x-y) for x,y in zip(car_count_list, body_count_list)]
# dist_body_mariadb = [np.linalg.norm(x-y) for x,y in zip(body_count_list, mariadb_counts_list)]
# dist_mariadb_mysql = [np.linalg.norm(x-y) for x,y in zip(mariadb_counts_list, mysql_counts_list)]
# dist_mysql_postgres = [np.linalg.norm(x-y) for x,y in zip(mysql_counts_list, postgres_counts_list)]
# dist_postgres_sysbench = [np.linalg.norm(x-y) for x,y in zip(postgres_counts_list, sysbench_counts_list)]
#
# print(dist_car_body)
# print(dist_body_mariadb)
# print(dist_mariadb_mysql)
# print(dist_mysql_postgres)
# print(dist_postgres_sysbench)

wcss = []
list_x = []

for i in range(1,10, 1):
    km = KMeans(n_clusters=i)
    km.fit_predict(normalized_df)
    wcss.append(km.inertia_)
    list_x.append(i)

print(wcss)
print(list_x)
#plt.plot(list_x, wcss)
#plt.show()

centroid_predictions = []

#clusters make 2,3 or 4
km = KMeans(n_clusters=4)
fitting = km.fit(normalized_df)
centroids_cluster = km.cluster_centers_
predictions = km.predict(normalized_df)
print(centroids_cluster)

centre0 = centroids_cluster[0]
centre1 = centroids_cluster[1]
centre2 = centroids_cluster[2]
centre3 = centroids_cluster[3]
#print(centre1)

centroid_labels = [centroids_cluster[i] for i in predictions]
#print(centroid_labels[0])
#print(type(centroid_labels))

dict_clusters_and_numbers = {0:0,1:0,2:0,3:0,4:0}
list_cluster_and_numbers = []

list_of_actual_values =normalized_df.values.tolist()
print(list_of_actual_values)
#with actual labels and clusters
for iterator1 in range(len(list_of_actual_values)):
    values = list_of_actual_values[iterator1]
    dist0 = float([np.linalg.norm(x - y) for x, y in zip(values, centre0)][0])
    dist1 = float([np.linalg.norm(x - y) for x, y in zip(values, centre1)][0])
    dist2 = float([np.linalg.norm(x - y) for x, y in zip(values, centre2)][0])
    dist3 = float([np.linalg.norm(x - y) for x, y in zip(values, centre3)][0])
    #print(values, centre0)
    #print(dist0)
    #print(type(dist0))
    #exit()
    #dist0 = format(dist0, ".2f")
    #dist1 = format(dist1, ".2f")
    #dist2 = format(dist2, ".2f")
    #print(type(dist0))
    if math.isclose(dist0, 0.0):
        dict_clusters_and_numbers[0] +=1
    elif math.isclose(dist1, 0.0):
        dict_clusters_and_numbers[1] +=1
    elif math.isclose(dist2, 0.0):
        dict_clusters_and_numbers[2] +=1
    elif math.isclose(dist3, 0.0):
        dict_clusters_and_numbers[3] +=1
    else:
        dict_clusters_and_numbers[4] += 1


print(dict_clusters_and_numbers)


exit()
#with predicted labels and clusters
for iterator1 in range(len(centroid_labels)):
    values = centroid_labels[iterator1]
    dist0 = float([np.linalg.norm(x - y) for x, y in zip(values, centre0)][0])
    dist1 = float([np.linalg.norm(x - y) for x, y in zip(values, centre1)][0])
    dist2 = float([np.linalg.norm(x - y) for x, y in zip(values, centre2)][0])
    dist3 = float([np.linalg.norm(x - y) for x, y in zip(values, centre3)][0])
    #print(values, centre0)
    print(dist0)
    print(type(dist0))
    #exit()
    #dist0 = format(dist0, ".2f")
    #dist1 = format(dist1, ".2f")
    #dist2 = format(dist2, ".2f")
    #print(type(dist0))
    if math.isclose(dist0, 0.0):
        dict_clusters_and_numbers[0] +=1
    elif math.isclose(dist1, 0.0):
        dict_clusters_and_numbers[1] +=1
    elif math.isclose(dist2, 0.0):
        dict_clusters_and_numbers[2] +=1
    elif math.isclose(dist3, 0.0):
        dict_clusters_and_numbers[3] +=1
    else:
        dict_clusters_and_numbers[4] += 1


print(dict_clusters_and_numbers)




exit()

dist_y_pred = np.array(dist)
print(np.percentile(dist, 99))
list_of_predictors = []

begin_iteration = 0
for begin_iteration in range(len(dist_y_pred)):
    if dist_y_pred[begin_iteration] >= np.percentile(dist, 90):
        list_of_predictors.append(0)
    else:
        list_of_predictors.append(1)

dist_y_pred[dist >= np.percentile(dist, 90)] = 0
dist_y_pred[dist <  np.percentile(dist, 90)] = 1








#Removing rows with all zeroes

# for iterator in range(len(system_call_list)):
#     if car_count_list[iterator] == 0 and body_count_list[iterator] == 0 and mariadb_counts_list[iterator] == 0 and mysql_counts_list[iterator] == 0 and postgres_counts_list[iterator] == 0 and sysbench_counts_list[iterator] ==0:
#         continue
#     else:
#         non_zero_system_call_list.append(system_call_list[iterator])
#         non_zero_car_count_list.append(car_count_list[iterator])
#         non_zero_body_count_list.append(body_count_list[iterator])
#         non_zero_mariadb_counts_list.append(mariadb_counts_list[iterator])
#         non_zero_mysql_counts_list.append(mysql_counts_list[iterator])
#         non_zero_postgres_counts_list.append(postgres_counts_list[iterator])
#         non_zero_sysbench_counts_list.append(sysbench_counts_list[iterator])
#
#
#
#
# new_df = pd.DataFrame(list(zip(non_zero_system_call_list
#                                ,non_zero_car_count_list
#                                ,non_zero_body_count_list
#                                ,non_zero_mariadb_counts_list
#                                ,non_zero_mysql_counts_list
#                                ,non_zero_postgres_counts_list
#                                ,non_zero_sysbench_counts_list))
#                       , columns=['system_calls', 'car_counts', 'body_counts', 'mariadb_counts', 'mysql_counts'
#                                  , 'postgres_counts', 'sysbench_counts'])

#print(new_df)

#new_df.to_csv('compared_555_removed_zeroes.csv', sep=',')

