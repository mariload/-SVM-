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

# 加载数据
def readdata(dataname):
    fr = open(dataname, 'r')
    all_lines = fr.readlines()
    testset = []
    for line in all_lines[0:]:
        if "?" in line:
            continue
        line = line.strip().split(',')  # 以逗号为分割符拆分列表
        testset.append(line)    
    data=np.array(testset)
    x = data[:, :-1].astype("float")
    y = data[:, -1].astype("float")
    print(x.shape, x.dtype)
    print(y.shape, y.dtype)
    train_x, test_x, train_y, test_y = ms.train_test_split(x, y, test_size=0.25, random_state=7)
    return train_x, test_x, train_y, test_y,x,y
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
        #result["precision"].append(recall)
        #result["f1"].append(recall)
        print(model.kernel,':',pred_y)
    return pd.DataFrame(result)
train_x, test_x, train_y, test_y,X, Y=readdata('purePCAdata.csv')
model = svm.SVC(C=100,kernel='linear', decision_function_shape="ovr")
model.fit(train_x, train_y)
predict_y = model.predict(test_x)

#cv=10采取10折交叉验证
train_sizes,train_scores,test_scores=learning_curve(model,X=X,y=Y,train_sizes=np.linspace(0.1,1.0,4),cv=10)
#统计结果
train_mean= np.mean(train_scores,axis=1)
train_std = np.std(train_scores,axis=1)
test_mean =np.mean(test_scores,axis=1)
test_std=np.std(test_scores,axis=1)
#绘制效果
plt.plot(train_sizes,train_mean,color='blue',marker='o',markersize=5,label='training accuracy')
plt.fill_between(train_sizes,train_mean+train_std,train_mean-train_std,alpha=0.15,color='blue')
plt.plot(train_sizes,test_mean,color='green',linestyle='--',marker='s',markersize=5,label='Cross-validation')
plt.fill_between(train_sizes,test_mean+test_std,test_mean-test_std,alpha=0.15,color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
#plt.ylim([0.8,1.0])
plt.show()






