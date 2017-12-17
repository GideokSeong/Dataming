import sys
from numpy import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn import neighbors, tree, naive_bayes
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("C:/Users/1/Desktop/group8/loan.csv", low_memory=False)

# 5% of the data without replacement
df = df.sample(frac=0.05, replace=False, random_state=123)
#print(df)
#print(df.shape)


df.head(n=5)
#print(df.columns)
#print(pd.unique(df['loan_status'].values.ravel()))
#print("Amount of Classes: ", len(pd.unique(df['loan_status'].values.ravel())))

#for col in df.select_dtypes(include=['object']).columns:
#    print ("Column {} has {} unique instances".format( col, len(df[col].unique())) )

print(" ")
#print(len(pd.unique(df['member_id'].values.ravel())) == df.shape[0])

df = df.drop('id', 1) #
df = df.drop('member_id', 1)#
df = df.drop('url', 1)#
df = df.drop('purpose', 1)
df = df.drop('title', 1)#
df = df.drop('zip_code', 1)#
df = df.drop('emp_title', 1)#
df = df.drop('earliest_cr_line', 1)#
df = df.drop('term', 1)
df = df.drop('sub_grade', 1) #
df = df.drop('last_pymnt_d', 1)#
df = df.drop('next_pymnt_d', 1)#
df = df.drop('last_credit_pull_d', 1)
df = df.drop('issue_d', 1) ##
df = df.drop('desc', 1)##
df = df.drop('addr_state', 1)##

#print(df.shape)

#for col in df.select_dtypes(include=['object']).columns:
#    print ("Column {} has {} unique instances".format( col, len(df[col].unique())) )
#df['loan_amnt'].plot(kind="hist", bins=10)
#df['grade'].value_counts().plot(kind='bar')
#df['emp_length'].value_counts().plot(kind='bar')
df['loan_status'].value_counts().plot(kind='bar')

plt.show()

#print(df._get_numeric_data().columns)

#print("There are {} numeric columns in the data set".format(len(df._get_numeric_data().columns) ))

#print(df.select_dtypes(include=['object']).columns)
#print("There are {} Character columns in the data set (minus the target)"
#.format(len(df.select_dtypes(include=['object']).columns) -1))


X = df.drop("loan_status", axis=1, inplace = False)
y = df.loan_status
#print(y.head())



def model_matrix(tf , columns):
    dummified_cols = pd.get_dummies(tf[columns])
    tf = tf.drop(columns, axis = 1, inplace=False)
    tf_new = tf.join(dummified_cols)
    return tf_new

X = model_matrix(X, ['grade', 'emp_length', 'home_ownership', 'verification_status',
                    'pymnt_plan', 'initial_list_status', 'application_type', 'verification_status_joint'])

#print(X.head())
#print(X.shape)

X2 = X.fillna(value = 0)
#print(X2.head())



Scaler = MinMaxScaler()

X2[['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate',
       'installment', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths',
       'mths_since_last_delinq', 'mths_since_last_record', 'open_acc',
       'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'out_prncp',
       'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp',
       'total_rec_int', 'total_rec_late_fee', 'recoveries',
       'collection_recovery_fee', 'last_pymnt_amnt',
       'collections_12_mths_ex_med', 'mths_since_last_major_derog',
       'policy_code', 'annual_inc_joint', 'dti_joint', 'acc_now_delinq',
       'tot_coll_amt', 'tot_cur_bal', 'open_acc_6m', 'open_il_6m',
       'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il',
       'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util',
       'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m']]= Scaler.fit_transform(X2[['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate',
       'installment', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths',
       'mths_since_last_delinq', 'mths_since_last_record', 'open_acc',
       'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'out_prncp',
       'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp',
       'total_rec_int', 'total_rec_late_fee', 'recoveries',
       'collection_recovery_fee', 'last_pymnt_amnt',
       'collections_12_mths_ex_med', 'mths_since_last_major_derog',
       'policy_code', 'annual_inc_joint', 'dti_joint', 'acc_now_delinq',
       'tot_coll_amt', 'tot_cur_bal', 'open_acc_6m', 'open_il_6m',
       'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il',
       'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util',
       'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m']])

#print(X2.head())

x_train, x_test, y_train, y_test = train_test_split(X2, y, test_size=.3, random_state=123)

#print(x_train.shape)
#print(y_train.shape)
#print(x_test.shape)
#print(y_test.shape)

data_knn = KNeighborsClassifier(n_neighbors = 10, metric='euclidean')
data_knn.fit(x_train, y_train)
#data_knn.predict(x_test)

rsquared_train = data_knn.score(x_train, y_train)
rsquared_test = data_knn.score(x_test, y_test)
#print ('Training data R-squared:')
#print(rsquared_train)
#print ('Test data R-squared:')
#print(rsquared_test)

knn_confusion_matrix = confusion_matrix(y_true = y_test, y_pred = data_knn.predict(x_test))
#print("The Confusion matrix:\n", knn_confusion_matrix)

"""
plt.matshow(knn_confusion_matrix, cmap = plt.cm.Blues)
plt.title("KNN Confusion Matrix\n")
#plt.xticks([0,1], ['No', 'Yes'])
#plt.yticks([0,1], ['No', 'Yes'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
for y in range(knn_confusion_matrix.shape[0]):
    for x in range(knn_confusion_matrix.shape[1]):
        plt.text(x, y, '{}'.format(knn_confusion_matrix[y, x]),
                horizontalalignment = 'center',
                verticalalignment = 'center',)
plt.show()
"""
knn_classify_report = classification_report(y_true = y_test,
                                           y_pred = data_knn.predict(x_test))
print(knn_classify_report)
