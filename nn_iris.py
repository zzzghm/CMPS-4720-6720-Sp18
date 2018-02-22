'''
we want to use panda to read the file
we randomly permuated the data; split the data into data train and data test.
the split is 70/30 for training and testing each
for convience, we manually parsed the data and saved them into two different files.
'''
import pandas as pd
"using panda package to read the files"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
torch.manual_seed(1234)
"""sets the seed for generating random numbers. And returns a torch._C.Generator object."""


"""read the file"""
train = pd.read_csv('../documents/iris_train.csv')
test = pd.read_csv('../documents/iris_test.csv')

"""buile the matrix"""
train.loc[train['species']=='Iris-setosa',['species']]=0
train.loc[train['species']=='Iris-versicolor',['species']]=1
train.loc[train['species']=='Iris-virginica',['species']]=2

"""build test matrix"""
test.loc[test['species']=='Iris-setosa',['species']]=0
test.loc[test['species']=='Iris-versicolor',['species']]=1
test.loc[test['species']=='Iris-virginica',['species']]=2


"""converting data into numeric"""
train = train.apply(pd.to_numeric)
test = test.apply(pd.to_numeric)

"""
print(train)
"""
"""change the frame to array"""
train_array = train.as_matrix()
test_array = test.as_matrix()

x_t = train_array[:,:4]
y_t = train_array[:,4]

x_test = test_array[:,:4]
y_test = test_array[:,4]

"""
print(x_t, y_t)
"""

"""
training
multilayer neural network, with one hidden layer.
input layer : 4 neuron, represents the feature of Iris
hidden layer : 15 neuron, activation using ReLU
hidden layer2 : 8 neuron, activation using ReLU
output layer : 3 neuron, represents the class of Iris
optimizer = stochastic gradient descent 
loss function = negative log likelihood class
learning rate = 0.012
epoch = 1000

"""

#setting params

hidden_layer = 15
hidden_layer2 = 8
learning_rate = 0.012
num_epoch = 1000

#build the model
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(4, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, hidden_layer2)
        self.fc3 = nn.Linear(hidden_layer2, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net()
print(net)        

#choose optimizer and loss function
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

#training
for epoch in range(num_epoch):
    X = Variable(torch.Tensor(x_t).float())
    Y = Variable(torch.Tensor(y_t).long())
    
    optimizer.zero_grad()
    out = net(X)
    loss = loss_func(out, Y)
    loss.backward()
    optimizer.step()
    if (epoch) % 20 == 0:
        print ('Epoch [%d/%d] Loss: %.4f' 
                   %(epoch+1, num_epoch, loss.data[0]))
"""
testing and get prediction 
"""
X = Variable(torch.Tensor(x_test).float())
Y = torch.Tensor(y_test).long()
out = net(X)
_, predicted = torch.max(out.data, 1)

#get accuration
print('Accuracy of the network %d %%' % (100 * torch.sum(Y==predicted) / 45))

