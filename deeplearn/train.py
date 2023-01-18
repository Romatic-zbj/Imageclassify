import torch
from sklearn import model_selection
import torch.nn as nn
import numpy as np
import pandas as pd

#data
trainfeatures=np.load("./dataset/train_features.npy")
print(trainfeatures.shape)
trainlabel=np.load("./dataset/train_labels.npy")
testfeatures=np.load("./dataset/test_features.npy")
print(trainlabel.shape)
Y=trainlabel
X=trainfeatures
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, random_state=1, test_size=0.3)
x_train=torch.from_numpy(x_train)
y_train=torch.from_numpy(y_train)
x_test=torch.from_numpy(x_test)
y_test=torch.from_numpy(y_test)


#net
class Net(torch.nn.Module):
    def __init__(self,n_feature,n_output):
        super(Net,self).__init__()
        self.layer1=torch.nn.Linear(n_feature,768)
        self.layer2=torch.nn.Linear(768,192)
        self.predict=torch.nn.Linear(192,n_output)

    def forward(self,x):
        out=self.layer1(x)
        out=torch.relu(out)
        out=self.layer2(out)
        out = torch.relu(out)
        out=self.predict(out)
        return out
net=Net(1536,20)

# model=nn.Sequential(
#     nn.Linear(1536,768),
#     nn.ReLU(),
#
#     nn.Linear(768,192),
#     nn.ReLU(),
#
#     nn.Linear(192,20),
#     nn.ReLU(),
# )
# print(model(x_train).shape)
#loss
loss_func=torch.nn.CrossEntropyLoss()

#optimizer

optimizer=torch.optim.SGD(net.parameters(),lr=0.1)

#train

for i in range(200):
    pred=net.forward(x_train)
    # print(pred.shape)
    # print(y_train.shape)
    loss=loss_func(pred,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    prediction = torch.max(pred, 1)[1]
    pred_y = prediction.data.numpy()
    target_y = y_train.long().data.numpy()
    accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
    # print(loss)

    #test
    predtest=net.forward(x_test)
    losstest=loss_func(predtest,y_test)
    predictiontest = torch.max(predtest, 1)[1]
    pred_test_y = predictiontest.data.numpy()
    target_test_y = y_test.long().data.numpy()
    accuracy_test = float((pred_test_y == target_test_y).astype(int).sum()) / float(target_test_y.size)
    if(i%5==0):
        print("epoch:{}\ttrain loss:{:.4f}\ttest loss:{:.4f}"
              "\tt_acc{:.4f}\tte_acc{:.4f}".format(i,loss,losstest,accuracy,accuracy_test))


#save model
torch.save(net,"model.pkl")

