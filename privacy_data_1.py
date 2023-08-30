import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer


non_zero_system_call_list = []
non_zero_car_count_list = []
non_zero_body_count_list = []
non_zero_mariadb_counts_list = []
non_zero_mysql_counts_list = []
non_zero_postgres_counts_list = []
non_zero_sysbench_counts_list = []


df = pd.read_csv ('compared_555.csv')

system_call_list     = df['system_call'].to_list()
car_count_list       = df['car_count'].to_list()
body_count_list      = df['body_count'].to_list()
mariadb_counts_list  = df['mariadb_counts'].to_list()
mysql_counts_list    = df['mysql_counts'].to_list()
postgres_counts_list = df['postgres_counts'].to_list()
sysbench_counts_list = df['sysbench_counts'].to_list()


#sc = StandardScaler()
sc = Normalizer()
car_count_list = sc.fit_transform(car_count_list)
body_count_list = sc.fit_transform(body_count_list)


dist = [np.linalg.norm(x-y) for x,y in zip(car_count_list, body_count_list)]

print(dist)

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

