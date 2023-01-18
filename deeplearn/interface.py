import torch
import numpy as np
import pandas as pd
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
#data
testfeatures=np.load("./dataset/test_features.npy")
testfeatures=torch.from_numpy(testfeatures).float()

#加载网络
net=torch.load("model.pkl")

#推理
prediction=net.forward(testfeatures)
predictiontest = torch.max(prediction, 1)[1]
#print(predictiontest.shape)
prediction=predictiontest.numpy()
print(prediction.shape)

output = pd.DataFrame({"Category": prediction})
output.index.name="Id"
output.to_csv('submission_deep.csv')

