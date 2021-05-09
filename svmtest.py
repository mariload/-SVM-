# coding:utf-8
import xlrd
import numpy as np
import sklearn.naive_bayes as nb
import matplotlib.pyplot as mp
import sklearn.svm as svm
import sklearn.model_selection as ms
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score,precision_score,f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve   #可视化学习的整个过程
from sklearn.model_selection import cross_val_score  #交叉验证
import operator
import pandas as pd
'''
    对SVM分类器进行参数调节，确定合适的惩罚系数C
    确定使用的核函数类型
'''
# 加载数据
def readdata(dataname,sheetname):
    wb = xlrd.open_workbook(dataname)
    #按工作簿定位工作表
    sh = wb.sheet_by_name(sheetname)
    testset = []
    for i in range(1,sh.nrows):
        line=sh.row_values(i)[3:]
        testset.append(line)  
    data=np.array(testset)
    x = data[:, :-1].astype("float")
    y = data[:, -1].astype("float")
    print(x.shape, x.dtype)
    print(y.shape, y.dtype)
    train_x, test_x, train_y, test_y = ms.train_test_split(x, y, test_size=0.25, random_state=7)
    return train_x, test_x, train_y, test_y

def svc_model(model):
    model.fit(train_x, train_y)
    acu_train = model.score(train_x, train_y)
    acu_test = model.score(test_x, test_y)
    pred_y = model.predict(test_x)
    recall = recall_score(test_y, pred_y, average="macro")    
    #precision = precision_score(test_y, pred_y,average='weighted')    
    #f1 = f1_score(test_y, pred_y,average='weighted')    
    return acu_train, acu_test, recall, pred_y
def svc(kernel,c):
    return svm.SVC(C=c,kernel=kernel, decision_function_shape="ovr")
   
def modelist(c):
    modelist = []
    kernalist = {"linear", "poly", "rbf", "sigmoid"}
    for each in kernalist:
        modelist.append(svc(each,c))
    return modelist
def run_svc_model(modelist):
    result = {"kernel": [],
              "acu_train": [],
              "acu_test": [],
              "recall": [],
              }
    for model in modelist:
        acu_train, acu_test, recall ,pred_y= svc_model(model)
        try:
            result["kernel"].append(model.kernel)
        except:
            result["kernel"].append(None)
        result["acu_train"].append(acu_train)
        result["acu_test"].append(acu_test)
        result["recall"].append(recall)
        print(model.kernel,':',pred_y)
    return pd.DataFrame(result)
train_x, test_x, train_y, test_y=readdata('处理后数据20210426.xlsx','x')
'''
for c in range(1,82,20):
    print("惩罚系数C:",c)
    dataframe=run_svc_model(modelist(c))
    print(dataframe)
'''
c=100
print("惩罚系数C:",c)
dataframe=run_svc_model(modelist(c))
print(dataframe)





