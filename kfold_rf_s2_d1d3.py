import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

#%matplotlib inline
matplotlib.rcParams.update({'font.size': 20})


supervised_path = 'train_test_supervised_with_timestamp/'
train_path = 'train_test_supervised_with_timestamp/'
test_path = 'train_test_supervised_with_timestamp/'
file1 = open('apps-sok-first-part.txt', 'r')
Lines1 = file1.readlines()
file2 = open('apps-sok-third-part.txt', 'r')
Lines2 = file2.readlines()
list_of_train = [1,2,3,4]
list_of_test =  [1,2,3,4]

def compute_roc_auc(x_values, y_values):
    #y_predict = rfc.predict_proba(x_values)[:,1]
    y_predict = rfc.predict(x_values)
    print(y_predict)
    fpr, tpr, thresholds = roc_curve(y_values, y_predict)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score

df_train_all = pd.DataFrame()
df_test_all  = pd.DataFrame()

predicted_list = []
fprs, tprs, scores = [], [], []


#Lines = file1.readlines()
#Lines1 = ['train_autoencoder_with_timestamp/train_autoencoder_all.pkl']
#Lines2 = ['test_autoencoder_with_timestamp/test_autoencoder_all.pkl']
sc = StandardScaler()
norm = MinMaxScaler()


fprs, tprs, scores = [], [], []

train_count=0

for test_element in range (1, 5):
    print(test_element)

    for line1 in Lines1:

        content = line1.strip()
        # print(content)
        for k in list_of_train:
            if k == test_element:
                continue

            else:
                path = train_path + content + '-' + str(k) + '.pkl'
                # path = 'C:/Users/sadun/PycharmProjects/cdl_remade/venv/data-CDL/' + content + '/' + content + '-' + str(k) + '_freqvector_full.csv'
                print("TRAIN")
                print(path)
                picklefile_train = open(path, 'rb')
                df_individual_train = pickle.load(picklefile_train)
                picklefile_train.close()
                df_train_all = pd.concat([df_train_all, df_individual_train], axis=0)


        train_count += 1

    for line2 in Lines2:

        content = line2.strip()

        for ki in list_of_test:
            if ki == test_element:
                path = test_path + content + '-' + str(ki) + '.pkl'
                # path = 'C:/Users/sadun/PycharmProjects/cdl_remade/venv/data-CDL/' + content + '/' + content + '-' + str(k) + '_freqvector_full.csv'
                print("TEST")
                print(path)
                picklefile_test = open(path, 'rb')
                df_individual_test = pickle.load(picklefile_test)
                picklefile_test.close()
                df_test_all = pd.concat([df_test_all, df_individual_test], axis=0)

            else:
                continue

    print(df_train_all.shape)
    print(df_test_all.shape)

    # df_train_all = pd.read_pickle(Lines1[0])
    train_data_x = df_train_all.iloc[:, :-1]
    train_data_y = df_train_all.iloc[:, -1:]

    # df_test_all  = pd.read_pickle(Lines2[0])
    test_data_x = df_test_all.iloc[:, :-1]
    test_data_y = df_test_all.iloc[:, -1:]
    # dataset_test = (test_x - min_val) / (max_val - min_val)
    # dataset_test = tf.cast(dataset_test, tf.float32)

    print(df_train_all.shape)
    print(df_test_all.shape)

    scaled_train_data_x = sc.fit_transform(train_data_x)
    scaled_test_data_x = sc.fit_transform(test_data_x)

    rfc = RandomForestClassifier(n_estimators=100, max_features=50, class_weight='balanced', n_jobs=-1)
    parameters = {
        "n_estimators": [200, 200, 200, 200, 200],
        "max_features": [100, 200, 300, 400, 500]
    }



    rfc.fit(scaled_train_data_x, train_data_y.values.ravel())
    _, _, auc_score_train = compute_roc_auc(scaled_train_data_x, train_data_y.values.ravel())
    fpr, tpr, auc_score = compute_roc_auc(scaled_test_data_x, test_data_y)
    scores.append((auc_score_train, auc_score))
    fprs.append(fpr)
    tprs.append(tpr)
    #scaled_test_data_x = X.iloc[test]
    #test_data_y = y.iloc[test].values.ravel()
    y_pred = rfc.predict(scaled_test_data_x)





    # y_pred=predictions
    # y_test=dataset_test
    # y_pred=np.argmax(predictions, axis=1)
    # y_test=np.argmax(dataset_test, axis=1)
    # print(y_pred)
    # print(y_test)
    # cm = confusion_matrix(y_test, y_pred)
    # print(cm)

    # fpr, tpr, thresholds = roc_curve(test_data_y, predictions)
    # auc_score = auc(fpr, tpr)
    # fprs.append(fpr)
    # tprs.append(tpr)

    print(confusion_matrix(test_data_y, y_pred))
    print(classification_report(test_data_y, y_pred))

    print("Accuracy :", accuracy_score(test_data_y, y_pred))
    # print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(test_data_y, y_pred, average='macro'))
    print("Recall   :", recall_score(test_data_y, y_pred, average='macro'))
    # Epoch 1, threshold 0.00287, accuracy 92.247
    score = f1_score(y_pred, test_data_y, average='macro')
    print('F-Measure: %.3f', score)


    total_number = 0
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    number_of_labels = len(test_data_y)
    test_labels = test_data_y
    lines_of_fpr = []
    lines_of_fnr = []

    #break


def plot_roc_curve_simple(fprs, tprs):
    plt.figure(figsize=(8,8))
    for i in range(len(fprs)):
        roc_auc = auc(fprs[i], tprs[i])
        ysmoothed = gaussian_filter1d(tprs[i], sigma=2)
        plt.plot(fprs[i], tprs[i], label='ROC fold %d (AUC = %0.2f)' % (i+1, roc_auc))
        i +=1
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for RF')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.savefig('RF_scenario_s2_d2d3.png',  bbox_inches='tight')
    plt.show()




plot_roc_curve_simple(fprs, tprs)

exit()


for i in range(len(test_labels)):
    if test_labels[i] == 1 and predicted_list[i] is True:
        true_positive +=1
        #print("TP")
    if test_labels[i] == 0 and predicted_list[i] is False:
        true_negative +=1
    if test_labels[i] == 0 and predicted_list[i] is True:
        false_negative +=1
        lines_of_fnr.append(i)
    if test_labels[i] == 1 and predicted_list[i] is False:
        false_positive +=1
        lines_of_fpr.append(i)

tpr = (true_positive/i)*100
tnr = (true_negative/i)*100
fpr = (false_positive/i)*100
fnr = (false_negative/i)*100

print("FALSE NEGATIVE:", false_negative)
print("FALSE POSITIVE:", false_positive)
print("TPR:", tpr)
print("TNR:", tnr)
print("FPR:", fpr)
print("FNR:", fnr)
#print(lines_of_fpr)
#print(lines_of_fnr)
