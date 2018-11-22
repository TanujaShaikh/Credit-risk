# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 23:21:40 2018

@author: tanuja
"""
#%%
import pandas as pd
import numpy as np
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
#%%

#%%
XYZ_DF=pd.read_csv(r'C:\Users\tanuja\OneDrive\Documents\imarticus\Group project\XYZCorp_LendingData.txt',header=0 ,
                      delimiter="\t", low_memory=False)

XYZ_DF.shape   #(855969, 73)
XYZ_DF.head()
#%%

#%%
#Creating copy of original data frame to go ahead for data preprocessing
XYZ_DF_rev=pd.DataFrame.copy(XYZ_DF)
XYZ_DF_rev.shape #(855969, 73)

#%%

#%% Feature Selection
# Out of 73 , few variables are not helpful or impactful in order to build a predictive model, hence dropping
XYZ_DF_rev.drop(['id','member_id','funded_amnt_inv','grade','emp_title','pymnt_plan','desc','title',
            'inq_last_6mths','mths_since_last_record','initial_list_status','mths_since_last_major_derog','policy_code',
            'dti_joint','verification_status_joint','tot_coll_amt','tot_cur_bal','open_acc_6m','open_il_6m','open_il_12m'
            ,'open_il_24m','mths_since_rcnt_il','total_bal_il','il_util','open_rv_12m','open_rv_24m',
            'max_bal_bc','all_util','inq_fi','total_cu_tl','inq_last_12m'],axis=1,inplace=True)

XYZ_DF_rev.shape #(855969, 42)
print(XYZ_DF_rev.head())

XYZ_DF_rev1=pd.DataFrame.copy(XYZ_DF_rev)
#%%

#%%
# Checking if missing values are present.
XYZ_DF_rev.isnull().sum()
print(XYZ_DF_rev.dtypes)
#%%

#%%
# Imputing categorical missing data with mode value

colname1=['term','sub_grade','emp_length','home_ownership','verification_status',
          'issue_d','purpose','zip_code','earliest_cr_line','last_pymnt_d',
          'next_pymnt_d','last_credit_pull_d','addr_state']
for x in colname1[:]:
    XYZ_DF_rev[x].fillna(XYZ_DF_rev[x].mode()[0],inplace=True)
    
XYZ_DF_rev.isnull().sum()
#%%


#%%
#Imputing missing data for Numerical variable with mean value / Zeros
XYZ_DF_rev['annual_inc_joint'].fillna(0,inplace=True)

colname2=['mths_since_last_delinq','revol_util','collections_12_mths_ex_med',
          'total_rev_hi_lim']
for x in colname2[:]:
    XYZ_DF_rev[x].fillna(XYZ_DF_rev[x].mean(),inplace=True)
    
XYZ_DF_rev.isnull().sum()
XYZ_DF_rev.shape   #(855969, 42)
#%%

#%%
#Label Encoding - to label all categorical variable value with numeric value
#Label will get assigned in Ascending alphabetical of variable value

colname1=['term','sub_grade','emp_length','home_ownership','verification_status',
          'purpose','zip_code','earliest_cr_line','last_pymnt_d',
          'next_pymnt_d','last_credit_pull_d','application_type','addr_state']

from sklearn import preprocessing

le={}

for x in colname1:
     le[x]=preprocessing.LabelEncoder()

for x in colname1:
     XYZ_DF_rev[x]=le[x].fit_transform(XYZ_DF_rev[x])
     
XYZ_DF_rev.head()
#%%
     
#%%
#Train and Test split

XYZ_DF_rev.issue_d = pd.to_datetime(XYZ_DF_rev.issue_d)   #%y-%m-%d
XYZ_DF_rev1.issue_d = pd.to_datetime(XYZ_DF_rev1.issue_d)

#split data in train and test on the basis of issue_d
split_date = "2015-05-01"

XYZ_training = XYZ_DF_rev.loc[XYZ_DF_rev['issue_d'] <= split_date]
XYZ_training=XYZ_training.drop(['issue_d'],axis=1)
XYZ_training.head()
XYZ_training.shape    #(598978, 41)

XYZ_test = XYZ_DF_rev.loc[XYZ_DF_rev['issue_d'] > split_date]
XYZ_test=XYZ_test.drop(['issue_d'],axis=1)
XYZ_test.head()
XYZ_test.shape  #(256991, 41)

#Exporting Train and Test dataframes to a CSV file for future purpose
XYZ_training_rev1 = XYZ_DF_rev1.loc[XYZ_DF_rev1['issue_d'] <= split_date]
XYZ_training_rev1=XYZ_training_rev1.drop(['issue_d'],axis=1)
XYZ_training_rev1.shape    #(598978, 41)
XYZ_training_rev1.to_csv(r'C:\Users\tanuja\OneDrive\Documents\imarticus\Group project\Final_train_rev.csv')

XYZ_test_rev1 = XYZ_DF_rev1.loc[XYZ_DF_rev1['issue_d'] > split_date]
XYZ_test_rev1=XYZ_test_rev1.drop(['issue_d'],axis=1)
XYZ_test_rev1.shape  #(256991, 41)
XYZ_test_rev1.to_csv(r'C:\Users\tanuja\OneDrive\Documents\imarticus\Group project\Final_test_rev.csv')

#%%

#%%
#%%
#selecting X and Y

X_train=XYZ_training.values[:,:-1]
Y_train=XYZ_training.values[:,-1]
Y_train=Y_train.astype(int)
print(Y_train)

X_test=XYZ_test.values[:,:-1]
Y_test=XYZ_test.values[:,-1]
Y_test=Y_test.astype(int)
print(Y_test)
#%%

#%%
#all reg module includes in sklearn.linear_model
from sklearn.linear_model import LogisticRegression
#create a model
classifier=LogisticRegression()
colname=XYZ_DF_rev.columns[:]
#fitting training data to the model
classifier.fit(X_train,Y_train)
#prediction
Y_pred=classifier.predict(X_test)
#print(list(zip(Y_test,Y_pred)))

#________________________Checking predictions____________________________________
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
#confusion_matrix
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)       # Type 2(63) , Type 1(40)
#classification_report
print("Classification report: ")
print(classification_report(Y_test,Y_pred))
#accuracy_score
acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc)    # 0.999599207754
#%%

#%%
#predicting using the Decision_Tree_Classifier
from sklearn.tree import DecisionTreeClassifier

model_DecisionTree=DecisionTreeClassifier()
model_DecisionTree.fit(X_train,Y_train)

#fit the model on the data and predict the values
Y_pred=model_DecisionTree.predict(X_test)

#________________________Checking predictions____________________________________
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
#confusion_matrix
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)     # Type 2(9) , Type 1(5963)
#classification_report
print("Classification report: ")
print(classification_report(Y_test,Y_pred))
#accuracy_score
acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc)    #0.976761832126
#%%

#%%
#predicting using the AdaBoost_Classifier
from sklearn.ensemble import AdaBoostClassifier

model_AdaBoost=(AdaBoostClassifier(base_estimator=LogisticRegression(),n_estimators=10))

#base_estimator= specify algo
#n_estimators= by default 50
#fit the model on the data and predict the values
model_AdaBoost.fit(X_train,Y_train)

Y_pred=model_AdaBoost.predict(X_test)

#________________________Checking predictions____________________________________

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
#confusion_matrix
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)         # Type 2(64) , Type 1(3454)
#classification_report
print("Classification report: ")
print(classification_report(Y_test,Y_pred))
#accuracy_score
acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc)   #0.986318587032

#%%

#%%
#predicting using the Gradient_Boosting_Classifier
from sklearn.ensemble import GradientBoostingClassifier

model_GradientBoosting=GradientBoostingClassifier()
#model_GradientBoosting=DecisionTreeClassifier()

#fit the model on the data and predict the values
model_GradientBoosting.fit(X_train,Y_train)

Y_pred=model_GradientBoosting.predict(X_test)

#________________________Checking predictions____________________________________

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
#confusion_matrix
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)         # Type 2(10) , Type 1(55)
#classification_report
print("Classification report: ")
print(classification_report(Y_test,Y_pred))
#accuracy_score
acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc)      #0.999747072855

#%%
#Since Gradient Boosting Classifier give us the best prediction we will Append Y_Pred to Test Data
Y_pred_col=list(Y_pred)
XYZ_test_rev1["Y_predictions"]=Y_pred_col
XYZ_test_rev1.head()

XYZ_test_rev1.to_csv(r'C:\Users\tanuja\OneDrive\Documents\imarticus\Group project\test_Pred_rev.csv')
#%%