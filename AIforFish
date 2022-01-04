import pandas as pd # for data manipulation
import numpy as np # for data manipulation

from sklearn.model_selection import train_test_split # for splitting the data into train and test samples
from sklearn.metrics import classification_report # for model evaluation metrics
from sklearn import tree # for decision tree models

import plotly.express as px  # for data visualization
import plotly.graph_objects as go # for data visualization
import graphviz # for plotting decision tree graphs
import torch as T
import gym
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
#import matlplotlib.pyplot as plt
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

class Net(nn.Module):
  def __init__(self,input_shape,o_shape):
    super(Net,self).__init__()
    self.fc1 = nn.Linear(input_shape,32)
    self.fc2 = nn.Linear(32,64)
    self.fc3 = nn.Linear(64,1)
  def forward(self,x):
    x = T.relu(self.fc1(x))
    x = T.relu(self.fc2(x))
    x = T.sigmoid(self.fc3(x))
    return x

print("Enter the path for the Data")
path1=input()
print("Enter the path for the model")
path2=input()
#fish= pd.read_csv("C:\\Users\\Shreyan Banerjee\\Desktop\\Week8\\COMPLETE_FISH_DATA.csv")
c1=pd.read_csv(path1,header=None)
#del fish['Time']
# list1=[]
# for i in range(501):
#    list1.append(T.Tensor(fish.loc[i])) 

TEST=T.Tensor(c1.loc[0])
#list2=T.stack(list1)
#list2=list2[:,1:]
c=TEST
#print(c)
model=Net(148,1)  
model.load_state_dict(T.load(path2))
emplist=[]
# for i in range(1):
#     name=input()
#     emplist=T.FloatTensor(name)

#inputs=T.stack(emplist).flatten()
op=model.forward(c)
if op>0.5:
    print("Damage Warning!!!")
else:
    print("All Good!!!")
