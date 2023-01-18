#多分类模型使用KNN,决策树，高斯朴素贝叶斯,
import numpy as np
from sklearn import svm
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB #朴素贝叶斯
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
def svc(kernel):#“linear”，“ploy”
    return svm.SVC(kernel=kernel, decision_function_shape="ovo")
def nusvc():
    return svm.NuSVC(decision_function_shape="ovo")
def linearsvc():
    return svm.LinearSVC(multi_class="ovr")
import sklearn.svm as svm
import pandas as pd
trainfeatures=np.load("./dataset/train_features.npy")
print(trainfeatures.shape)
trainlabel=np.load("./dataset/train_labels.npy")
testfeatures=np.load("./dataset/test_features.npy")
print(trainlabel.shape)
print(testfeatures.shape)
Y=trainlabel
X=trainfeatures
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, random_state=1, test_size=0.4)
classifier = RandomForestClassifier()#改这里
classifier.fit(x_train, y_train)
print('高斯贝叶斯输出训练集的准确率为: %.2f' % classifier.score(x_train, y_train))
print('高斯贝叶斯输出测试集的准确率为: %.2f' % classifier.score(x_test, y_test))
testfeatures=pd.DataFrame(testfeatures,index=testfeatures[:,0])
predictions=classifier.predict(testfeatures)
output = pd.DataFrame({"Category": predictions})
output.index.name="Id"
output.to_csv('submission_rf.csv')