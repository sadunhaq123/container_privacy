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

dict_clusters_and_numbers_iterative = {}
dict_clusters_and_centers_iterative = {}
name_list = []
distance_list = []
dict_of_names = {}

#list1 = ['car', 'body', 'mariadb', 'mysql', 'postgres', 'mongo', 'redis', 'wordpress', 'caddy', 'httpd', 'sysbench']
#list2 = ['car', 'body', 'mariadb', 'mysql', 'postgres', 'mongo', 'redis', 'wordpress', 'caddy', 'httpd', 'sysbench']

list1 = ['mysql', 'mariadb',  'postgres', 'redis', 'mongo',  'couchbase', 'car', 'body', 'sysbench', 'c', 'java', 'python']
list2 = ['mysql', 'mariadb',  'postgres', 'redis', 'mongo',  'couchbase', 'car', 'body', 'sysbench', 'c', 'java', 'python']

list_1d = []
list_2d = []

#df = pd.read_csv ('compared_555.csv')
df = pd.read_csv ('all_exp_split_sorted_name_555_removed_zeroth_rows.csv')

system_call_list     = df['system_calls'].to_list()
car_counts_1_list       = df['car_counts_1'].to_list()
body_counts_1_list      = df['body_counts_1'].to_list()
mariadb_counts_1_list  = df['mariadb_counts_1'].to_list()
mysql_counts_1_list    = df['mysql_counts_1'].to_list()
postgres_counts_1_list = df['postgres_counts_1'].to_list()
mongo_counts_1_list = df['mongo_counts_1'].to_list()
redis_counts_1_list = df['redis_counts_1'].to_list()
sysbench_counts_1_list = df['sysbench_counts_1'].to_list()
couchbase_counts_1_list = df['couchbase_counts_1'].to_list()
c_counts_1_list = df['c_counts_1'].to_list()
java_counts_1_list = df['java_counts_1'].to_list()
python_counts_1_list = df['python_counts_1'].to_list()
car_counts_2_list       = df['car_counts_2'].to_list()
body_counts_2_list      = df['body_counts_2'].to_list()
mariadb_counts_2_list  = df['mariadb_counts_2'].to_list()
mysql_counts_2_list    = df['mysql_counts_2'].to_list()
postgres_counts_2_list = df['postgres_counts_2'].to_list()
mongo_counts_2_list = df['mongo_counts_2'].to_list()
redis_counts_2_list = df['redis_counts_2'].to_list()
sysbench_counts_2_list = df['sysbench_counts_2'].to_list()
couchbase_counts_2_list = df['couchbase_counts_2'].to_list()
c_counts_2_list = df['c_counts_2'].to_list()
java_counts_2_list = df['java_counts_2'].to_list()
python_counts_2_list = df['python_counts_2'].to_list()


#print(df)
sum_car_counts_1 = df['car_counts_1'].sum()
sum_body_counts_1 = df['body_counts_1'].sum()
sum_mariadb_counts_1 = df['mariadb_counts_1'].sum()
sum_mysql_counts_1 = df['mysql_counts_1'].sum()
sum_postgres_counts_1 = df['postgres_counts_1'].sum()
sum_mongo_counts_1 = df['mongo_counts_1'].sum()
sum_redis_counts_1 = df['redis_counts_1'].sum()
sum_sysbench_counts_1 = df['sysbench_counts_1'].sum()
sum_couchbase_counts_1 = df['couchbase_counts_1'].sum()
sum_c_counts_1 = df['c_counts_1'].sum()
sum_java_counts_1 = df['java_counts_1'].sum()
sum_python_counts_1 = df['python_counts_1'].sum()

sum_car_counts_2 = df['car_counts_2'].sum()
sum_body_counts_2 = df['body_counts_2'].sum()
sum_mariadb_counts_2 = df['mariadb_counts_2'].sum()
sum_mysql_counts_2 = df['mysql_counts_2'].sum()
sum_postgres_counts_2 = df['postgres_counts_2'].sum()
sum_mongo_counts_2 = df['mongo_counts_2'].sum()
sum_redis_counts_2 = df['redis_counts_2'].sum()
sum_sysbench_counts_2 = df['sysbench_counts_2'].sum()
sum_couchbase_counts_2 = df['couchbase_counts_2'].sum()
sum_c_counts_2 = df['c_counts_2'].sum()
sum_java_counts_2 = df['java_counts_2'].sum()
sum_python_counts_2 = df['python_counts_2'].sum()
column_sums = df.sum()
#print(sum_car_counts)
#df_result = df.divide(df / column_sums)


df_averaged = pd.DataFrame()
df_averaged['car_counts_1'] = df['car_counts_1']/sum_car_counts_1
df_averaged['car_counts_2'] = df['car_counts_2']/sum_car_counts_2
df_averaged['body_counts_1'] = df['body_counts_1']/sum_body_counts_1
df_averaged['body_counts_2'] = df['body_counts_2']/sum_body_counts_2
df_averaged['mariadb_counts_1'] = df['mariadb_counts_1']/sum_mariadb_counts_1
df_averaged['mariadb_counts_2'] = df['mariadb_counts_2']/sum_mariadb_counts_2
df_averaged['mysql_counts_1'] = df['mysql_counts_1']/sum_mysql_counts_1
df_averaged['mysql_counts_2'] = df['mysql_counts_2']/sum_mysql_counts_2
df_averaged['postgres_counts_1'] = df['postgres_counts_1']/sum_postgres_counts_1
df_averaged['postgres_counts_2'] = df['postgres_counts_2']/sum_postgres_counts_2
df_averaged['mongo_counts_1'] = df['mongo_counts_1']/sum_mongo_counts_1
df_averaged['mongo_counts_2'] = df['mongo_counts_2']/sum_mongo_counts_2
df_averaged['redis_counts_1'] = df['redis_counts_1']/sum_redis_counts_1
df_averaged['redis_counts_2'] = df['redis_counts_2']/sum_redis_counts_2
df_averaged['couchbase_counts_1'] = df['couchbase_counts_1']/sum_couchbase_counts_1
df_averaged['couchbase_counts_2'] = df['couchbase_counts_2']/sum_couchbase_counts_2
df_averaged['c_counts_1'] = df['c_counts_1']/sum_c_counts_1
df_averaged['c_counts_2'] = df['c_counts_2']/sum_c_counts_2
df_averaged['java_counts_1'] = df['java_counts_1']/sum_java_counts_1
df_averaged['java_counts_2'] = df['java_counts_2']/sum_java_counts_2
df_averaged['python_counts_1'] = df['python_counts_1']/sum_python_counts_1
df_averaged['python_counts_2'] = df['python_counts_2']/sum_python_counts_2
df_averaged['sysbench_counts_1'] = df['sysbench_counts_1']/sum_sysbench_counts_1
df_averaged['sysbench_counts_2'] = df['sysbench_counts_2']/sum_sysbench_counts_2

#print(df_averaged)
#exit()

car_counts_list_1 = df_averaged['car_counts_1'].to_list()
car_counts_list_2 = df_averaged['car_counts_2'].to_list()
body_counts_list_1 = df_averaged['body_counts_1'].to_list()
body_counts_list_2 = df_averaged['body_counts_2'].to_list()
mariadb_counts_list_1 = df_averaged['mariadb_counts_1'].to_list()
mariadb_counts_list_2 = df_averaged['mariadb_counts_2'].to_list()
mysql_counts_list_1 = df_averaged['mysql_counts_1'].to_list()
mysql_counts_list_2 = df_averaged['mysql_counts_2'].to_list()
postgres_counts_list_1 = df_averaged['postgres_counts_1'].to_list()
postgres_counts_list_2 = df_averaged['postgres_counts_2'].to_list()
mongo_counts_list_1 = df_averaged['mongo_counts_1'].to_list()
mongo_counts_list_2 = df_averaged['mongo_counts_2'].to_list()
redis_counts_list_1 = df_averaged['redis_counts_1'].to_list()
redis_counts_list_2 = df_averaged['redis_counts_2'].to_list()
couchbase_counts_list_1 = df_averaged['couchbase_counts_1'].to_list()
couchbase_counts_list_2 = df_averaged['couchbase_counts_2'].to_list()
c_counts_list_1 = df_averaged['c_counts_1'].to_list()
c_counts_list_2 = df_averaged['c_counts_2'].to_list()
java_counts_list_1 = df_averaged['java_counts_1'].to_list()
java_counts_list_2 = df_averaged['java_counts_2'].to_list()
python_counts_list_1 = df_averaged['python_counts_1'].to_list()
python_counts_list_2 = df_averaged['python_counts_2'].to_list()
sysbench_counts_list_1 = df_averaged['sysbench_counts_1'].to_list()
sysbench_counts_list_2 = df_averaged['sysbench_counts_2'].to_list()

#print(car_counts_list)

#column_sums = df_averaged.sum()
#print(column_sums)

#exit()



#sc = StandardScaler()
#sc = Normalizer()

#l1
# car_counts_list = normalize([car_counts_list], norm="l1").tolist()[0]
# body_counts_list = normalize([body_counts_list], norm="l1").tolist()[0]
# mariadb_counts_list = normalize([mariadb_counts_list], norm="l1").tolist()[0]
# mysql_counts_list = normalize([mysql_counts_list], norm="l1").tolist()[0]
# postgres_counts_list = normalize([postgres_counts_list], norm="l1").tolist()[0]
# mongo_counts_list = normalize([mongo_counts_list], norm="l1").tolist()[0]
# redis_counts_list = normalize([redis_counts_list], norm="l1").tolist()[0]
# sysbench_counts_list = normalize([sysbench_counts_list], norm="l1").tolist()[0]
# couchbase_counts_list = normalize([couchbase_counts_list], norm="l1").tolist()[0]
# c_counts_list = normalize([c_counts_list], norm="l1").tolist()[0]
# java_counts_list = normalize([java_counts_list], norm="l1").tolist()[0]
# python_counts_list = normalize([python_counts_list], norm="l1").tolist()[0]
# print(car_counts_list)
# print(body_counts_list)

#print(type(car_counts_list)).tolist()[0]

#print(car_counts_list.shape)
#l2
# car_counts_list = normalize([car_counts_list], norm="l2").tolist()[0]
# body_counts_list = normalize([body_counts_list], norm="l2").tolist()[0]
# mariadb_counts_list = normalize([mariadb_counts_list], norm="l2").tolist()[0]
# mysql_counts_list = normalize([mysql_counts_list], norm="l2").tolist()[0]
# postgres_counts_list = normalize([postgres_counts_list], norm="l2").tolist()[0]
# mongo_counts_list = normalize([mongo_counts_list], norm="l2").tolist()[0]
# redis_counts_list = normalize([redis_counts_list], norm="l2").tolist()[0]
# sysbench_counts_list = normalize([sysbench_counts_list], norm="l2").tolist()[0]
# couchbase_counts_list = normalize([couchbase_counts_list], norm="l2").tolist()[0]
# c_counts_list = normalize([c_counts_list], norm="l2").tolist()[0]
# java_counts_list = normalize([java_counts_list], norm="l2").tolist()[0]
# python_counts_list = normalize([python_counts_list], norm="l2").tolist()[0]
# print(car_counts_list)
# print(body_counts_list)

#max
# car_counts_list = normalize([car_counts_list], norm="max").tolist()[0]
# body_counts_list = normalize([body_counts_list], norm="max").tolist()[0]
# mariadb_counts_list = normalize([mariadb_counts_list], norm="max").tolist()[0]
# mysql_counts_list = normalize([mysql_counts_list], norm="max").tolist()[0]
# postgres_counts_list = normalize([postgres_counts_list], norm="max").tolist()[0]
# mongo_counts_list = normalize([mongo_counts_list], norm="max").tolist()[0]
# redis_counts_list = normalize([redis_counts_list], norm="max").tolist()[0]
# sysbench_counts_list = normalize([sysbench_counts_list], norm="max").tolist()[0]
# couchbase_counts_list = normalize([couchbase_counts_list], norm="max").tolist()[0]
# c_counts_list = normalize([c_counts_list], norm="max").tolist()[0]
# java_counts_list = normalize([java_counts_list], norm="max").tolist()[0]
# python_counts_list = normalize([python_counts_list], norm="max").tolist()[0]
# print(car_counts_list)
# print(body_counts_list)

#print(body_counts_list)
#normalized_df['car'] = pd.Series(car_counts_list)

normalized_df = pd.DataFrame(list(zip(car_counts_list_1, car_counts_list_2
                                      , body_counts_list_1, body_counts_list_2
                                      , mariadb_counts_list_1, mariadb_counts_list_2
                                      , mysql_counts_list_1, mysql_counts_list_2
                                      , postgres_counts_list_1, postgres_counts_list_2
                                      , mongo_counts_list_1, mongo_counts_list_2
                                      , redis_counts_list_1, redis_counts_list_2
                                      , couchbase_counts_list_1, couchbase_counts_list_2
                                      , c_counts_list_1, c_counts_list_2
                                      , java_counts_list_1, java_counts_list_2
                                      , python_counts_list_1, python_counts_list_2
                                      , sysbench_counts_list_1, sysbench_counts_list_2
                                      )))
#normalized_df.columns = ['car', 'body', 'mariadb', 'mysql', 'postgres', 'sysbench']
#normalized_df = normalized_df.T
#print(normalized_df)
#exit()
#print(normalized_df.take([5]))
#normalized_df = pd.concat([normalized_df, df_individual_train], axis=0)

# print("CPNMV:",np.array([car_counts_list]))
# print(np.array([car_counts_list]).shape)
# exit()

#print(normalized_df)

#exit()
df_abs_diff = pd.DataFrame()
df_abs_diff['system_calls'] = df['system_calls']
df_abs_diff['car'] = abs(df['car_counts_1'] - df['car_counts_2'])
df_abs_diff['body'] = abs(df['body_counts_1'] - df['body_counts_2'])
df_abs_diff['mysql'] = abs(df['mysql_counts_1'] - df['mysql_counts_2'])
df_abs_diff['mariadb'] = abs(df['mariadb_counts_1'] - df['mariadb_counts_2'])
df_abs_diff['couchbase'] = abs(df['couchbase_counts_1'] - df['couchbase_counts_2'])
df_abs_diff['postgres'] = abs(df['postgres_counts_1'] - df['postgres_counts_2'])
df_abs_diff['mongo'] = abs(df['mongo_counts_1'] - df['mongo_counts_2'])
df_abs_diff['redis'] = abs(df['redis_counts_1'] - df['redis_counts_2'])
df_abs_diff['sysbench'] = abs(df['sysbench_counts_1'] - df['sysbench_counts_2'])
df_abs_diff['c'] = abs(df['c_counts_1'] - df['c_counts_2'])
df_abs_diff['java'] = abs(df['java_counts_1'] - df['java_counts_2'])
df_abs_diff['python'] = abs(df['python_counts_1'] - df['python_counts_2'])
#df_abs_diff['car'] = abs(normalized_df[0]-normalized_df[1])

threshold = 10000
#print(df)
#print(df_abs_diff)
car_threshold = df_abs_diff[df_abs_diff['car'] > threshold]['car']
body_threshold = df_abs_diff[df_abs_diff['body'] > threshold]['body']
mariadb_threshold = df_abs_diff[df_abs_diff['mariadb'] > threshold]['mariadb']
mysql_threshold = df_abs_diff[df_abs_diff['mysql'] > threshold]['mysql']
redis_threshold = df_abs_diff[df_abs_diff['redis'] > threshold]['redis']
mongo_threshold = df_abs_diff[df_abs_diff['mongo'] > threshold]['mongo']
couchbase_threshold = df_abs_diff[df_abs_diff['couchbase'] > threshold]['couchbase']
postgres_threshold = df_abs_diff[df_abs_diff['postgres'] > threshold]['postgres']
sysbench_threshold = df_abs_diff[df_abs_diff['sysbench'] > threshold]['sysbench']
c_threshold = df_abs_diff[df_abs_diff['c'] > threshold]['c']
java_threshold = df_abs_diff[df_abs_diff['java'] > threshold]['java']
python_threshold = df_abs_diff[df_abs_diff['python'] > threshold]['python']
print("CAR:", car_threshold)
print("BODY:", body_threshold)
print("MARIADB:", mariadb_threshold)
print("MYSQL:", mysql_threshold)
print("REDIS:", redis_threshold)
print("MONGO:", mongo_threshold)
print("COUCHBASE:", couchbase_threshold)
print("POSTGRES:", postgres_threshold)
print("SYSBENCH:", sysbench_threshold)
print("C:", c_threshold)
print("JAVA:", java_threshold)
print("PYTHON:", python_threshold)
#c_a_b = car_threshold[['system_calls','car']]
#print(c_a_b)

exit()


dict_of_names['car_counts_list_1'] = np.array([car_counts_list_1])
dict_of_names['car_counts_list_2'] = np.array([car_counts_list_2])
dict_of_names['body_counts_list_1'] = np.array([body_counts_list_1])
dict_of_names['body_counts_list_2'] = np.array([body_counts_list_2])
dict_of_names['mariadb_counts_list_1'] = np.array([mariadb_counts_list_1])
dict_of_names['mariadb_counts_list_2'] = np.array([mariadb_counts_list_2])
dict_of_names['mysql_counts_list_1'] = np.array([mysql_counts_list_1])
dict_of_names['mysql_counts_list_2'] = np.array([mysql_counts_list_2])
dict_of_names['postgres_counts_list_1'] = np.array([postgres_counts_list_1])
dict_of_names['postgres_counts_list_2'] = np.array([postgres_counts_list_2])
dict_of_names['mongo_counts_list_1'] = np.array([mongo_counts_list_1])
dict_of_names['mongo_counts_list_2'] = np.array([mongo_counts_list_2])
dict_of_names['redis_counts_list_1'] = np.array([redis_counts_list_1])
dict_of_names['redis_counts_list_2'] = np.array([redis_counts_list_2])
dict_of_names['couchbase_counts_list_1'] = np.array([couchbase_counts_list_1])
dict_of_names['couchbase_counts_list_2'] = np.array([couchbase_counts_list_2])
dict_of_names['c_counts_list_1'] = np.array([c_counts_list_1])
dict_of_names['c_counts_list_2'] = np.array([c_counts_list_2])
dict_of_names['java_counts_list_1'] = np.array([java_counts_list_1])
dict_of_names['java_counts_list_w'] = np.array([java_counts_list_2])
dict_of_names['python_counts_list_1'] = np.array([python_counts_list_1])
dict_of_names['python_counts_list_2'] = np.array([python_counts_list_2])
dict_of_names['sysbench_counts_list_1'] = np.array([sysbench_counts_list_1])
dict_of_names['sysbench_counts_list_2'] = np.array([sysbench_counts_list_2])
#car_count_list = np.reshape(car_count_list,(-1,1))
#body_count_list = np.reshape(body_count_list,(-1,1))
#print("HELLO:",dict_of_names['car_counts_list'])

#exit()


#car_count_list = sc.fit_transform(car_count_list)
#body_count_list = sc.fit_transform(body_count_list)

#print(car_count_list)

l1 = []
l2 = []

# print(dict_of_names)
# exit()

# for iterator1 in list1:
#     for iterator2 in list2:
#         name = 'dist_' + iterator1 +'_' + iterator2
#         name1 = iterator1
#         name2 = iterator2
#         candidate_1 = iterator1 + '_counts_list'
#         candidate_2 = iterator2 + '_counts_list'
#         #print(candidate_1)
#         #print(candidate_2)
#         dist = [np.linalg.norm(x-y) for x,y in zip(dict_of_names[candidate_1], dict_of_names[candidate_2])][0]
#         dist = format(dist,".2f")
#         #print(name, dist)
#         if float(dist) < 0.25:
#             if dict_clusters_and_numbers_iterative.get(name1) is None:
#                 dict_clusters_and_numbers_iterative[name1] = []
#                 dict_clusters_and_centers_iterative[name1] = []
#             else:
#                 dict_clusters_and_numbers_iterative[name1].append(name2)
#                 dict_clusters_and_centers_iterative[name1].append(dist)
#
#         else:
#             if dict_clusters_and_numbers_iterative.get(name2) is None:
#                 dict_clusters_and_numbers_iterative[name2] = []
#                 dict_clusters_and_centers_iterative[name2] = []
#
#             # else:
#             #     dict_clusters_and_numbers_iterative[name1] = l1.append(name2)
#
#         name_list.append(name)
#         distance_list.append(dist)
#         list_1d.append(dist)
#     list_2d.append(list_1d)
#     list_1d = []

# print(dict_clusters_and_numbers_iterative)
# print(dict_clusters_and_centers_iterative)


for iterator1 in dict_of_names:
    for iterator2 in dict_of_names:
        name = 'dist_' + iterator1 +'_' + iterator2
        name1 = iterator1
        name2 = iterator2
        candidate_1 = iterator1
        candidate_2 = iterator2
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

#print(list1)
for i in list_2d:
   print(*i)



#my_df = pd.DataFrame(list_2d)
#my_df.to_csv('split_comparatar.csv', index=False)
exit()
# if threshold is < 0.25 we have 3 clusters
# 1. mysql, mariadb and redis
# 2. car, body
# 3. outliers postgres, mongo, couchbase, sysbench, c, java, python

centre_of_cluster_1 = np.mean([mysql_counts_list, mariadb_counts_list, redis_counts_list], axis=0)
centre_of_cluster_2 = np.mean([car_counts_list, body_counts_list], axis=0)
outliers = [postgres_counts_list,mongo_counts_list,sysbench_counts_list, couchbase_counts_list, c_counts_list, java_counts_list, python_counts_list]

#print(centre_of_cluster_1)
#print(outliers[0])

for iterator in outliers:
    #print(outliers[iterator])
    #exit()
    #print(type(centre_of_cluster_1))
    #print(type(np.array(iterator)))
    #print((centre_of_cluster_1.shape))
    #print((np.array(iterator).shape))
    distance_from_cluster1 = np.linalg.norm(centre_of_cluster_1 - np.array(iterator))
    distance_from_cluster2 = np.linalg.norm(centre_of_cluster_2 - np.array(iterator))
    print(distance_from_cluster1, distance_from_cluster2)


#print(centre_of_cluster_1)

# point1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# point2 = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
#
# print(point1)
# print(type(point1))
# print(point1.shape)
#
# # Calculate Euclidean distance
# distance = np.linalg.norm(point2 - point1)
#
# print("Euclidean Distance between the two points:")
# print(distance)

#dict_clusters_and_numbers_iterative

#exit()



#my_df.to_csv('max_csv.csv', index=False, header=False)
#exit()
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

#print(normalized_df)
normalized_df = normalized_df.transpose()
#print(normalized_df)
#exit()
for i in range(1,10, 1):
    km = KMeans(n_clusters=i)
    km.fit_predict(normalized_df)
    wcss.append(km.inertia_)
    list_x.append(i)

print(wcss)
print(list_x)
plt.plot(list_x, wcss)
plt.show()


#exit()
centroid_predictions = []

print("Normalized DF:", normalized_df.shape)
#print(normalized_df[0])
#clusters make 2,3 or 4
km = KMeans(n_clusters=6)
fitting = km.fit(normalized_df)
centroids_cluster = km.cluster_centers_
#print(type(centroids_cluster.shape))
#exit()
predictions = km.predict(normalized_df)

clusters = km.fit_predict(normalized_df)
#print(centroids_cluster)


distances = km.transform(normalized_df)
print("Clusters:",clusters)
print("No of elements:",clusters.size)
threshold = 3.0

new_clusters = [cluster if np.min(distances[i, :]) < threshold else k for i, cluster in enumerate(clusters)]
#print(new_clusters)

df = normalized_df
#print(df)
df['cluster'] = new_clusters
df['distances_from_centers'] = distances.tolist()




# Print the results
for cluster in df['cluster'].unique():
    cluster_points = df[df['cluster'] == cluster].drop('cluster', axis=1)
    print(f"Cluster {cluster}:")
    print(cluster_points)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


print("Distances of each data point from all cluster centers:")
print(df[['distances_from_centers', 'cluster']].to_markdown())


# cluster_points = {}
# for i, cluster in enumerate(new_clusters):
#     if cluster not in cluster_points:
#         cluster_points[cluster] = []md
#     cluster_points[cluster].append(normalized_df[i])
#
# # Print the results
# for cluster, points in cluster_points.items():
#     print(f"Cluster {cluster}:")
#     print(points)

#print(normalized_df)

exit()

centre0 = centroids_cluster[0]
centre1 = centroids_cluster[1]
centre2 = centroids_cluster[2]
centre3 = centroids_cluster[3]
centre4 = centroids_cluster[4]
centre5 = centroids_cluster[5]
centre6 = centroids_cluster[6]
#print(centre1)

centroid_labels = [centroids_cluster[i] for i in predictions]
#print(centroid_labels[0])
#print(type(centroid_labels))

dict_clusters_and_numbers = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0}
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
    dist4 = float([np.linalg.norm(x - y) for x, y in zip(values, centre4)][0])
    dist5 = float([np.linalg.norm(x - y) for x, y in zip(values, centre5)][0])
    dist6 = float([np.linalg.norm(x - y) for x, y in zip(values, centre6)][0])
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
    elif math.isclose(dist4, 0.0):
        dict_clusters_and_numbers[4] +=1
    elif math.isclose(dist5, 0.0):
        dict_clusters_and_numbers[5] +=1
    elif math.isclose(dist6, 0.0):
        dict_clusters_and_numbers[6] +=1
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
