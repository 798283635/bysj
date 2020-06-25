from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn as nn
import numpy as np
import torch


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import pandas as  pd

import matplotlib.pyplot as plt
torch.manual_seed(1)
look_back = 25
hidden_num = 128
batch = 128
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.lstm = nn.LSTM(
            input_size=1,  # 特征数
            hidden_size=hidden_num,  # rnn hidden unit
            num_layers=1,  # 有几层 RNN layers
            batch_first=True,
        )
        #self.out = nn.Sequential(nn.Linear(hidden_num, 64), nn.ReLU(), nn.Linear(64, 1), nn.ReLU())
        #self.out = nn.Linear(hidden_num, 1, nn.ReLU())
        self.out = nn.Sequential(nn.Linear(hidden_num,1),nn.LeakyReLU(0.2))
    def forward(self,x):

        r_out, (hn, cn) = self.lstm(x)

        #print(r_out.shape)#(batch, time_step, num_directions * hidden_size)

        outs = []  # 保存所有时间点的预测值
        for time_step in range(r_out.size(1)):  # 对每一个时间点计算 output
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1),  hn, cn


class  Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.out = nn.Sequential(  # Discriminator
            nn.Linear(look_back + 1, 72),  # receive art work either from the famous artist or a newbie like G
            nn.LeakyReLU(0.2),
            nn.Linear(72, 100),
            nn.LeakyReLU(0.2),
            nn.Linear(100, 10),
            nn.LeakyReLU(0.2),
            nn.Linear(10, 1),

        )


    def forward(self, x):
        return self.out(x)


def MaxMinNormalization(x):
    """[0,1] normaliaztion"""
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x


def ZscoreNormalization(x):
    """Z-score normaliaztion"""
    x = (x - np.mean(x)) / np.std(x)
    return x







dataframe = pd.read_csv('after_six1.csv', sep=',')
dataframe  = dataframe[0:1000]
dataset = dataframe["hourly_traffic_count"].values
# 将整型变为float
dataset = dataset.astype('float32')

dataset = ZscoreNormalization(dataset)



dataset = np.array(dataset)
dataset = dataset.reshape(-1,1)







def create_dataset(dataset, look_back):
#这里的look_back与timestep相同
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        #dataset[i:(i+look_back)]
        dataX.append(dataset[i:(i+look_back)])
        dataY.append(dataset[i + 1:(i+look_back + 1)])
    return np.array(dataX),np.array(dataY)
#训练数据太少 look_back并不能过大

data_X, data_Y = create_dataset(dataset,look_back)



train_X = data_X
train_Y = data_Y
train_X = train_X.reshape(-1, look_back, 1)
train_Y = train_Y.reshape(-1, look_back, 1)
train_x = torch.from_numpy(train_X)
train_y = torch.from_numpy(train_Y)


dataset = TensorDataset(train_x,train_y)
#dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)


model = Generator()

D =  Discriminator()

criterion = nn.MSELoss()


opt_D = torch.optim.Adam(D.parameters(), lr=0.0001)
opt_G = torch.optim.Adam(model.parameters(), lr=0.001)



def train_gan():

    for epoch in range(0,2000):
        y_predict = []
        cnt = 0
        print("epoch |", epoch + 1)
        model.train()
        for batch_x, batch_y in dataloader:
            cnt += 1
            y_fake = []
            y_real = []
            y_, h, c = model(batch_x)
            for i in range(len(batch_x)):

                #print("aclove",var_x[i],y[i][-1].view(-1,1))
                y_fake.append(torch.cat((batch_x[i],y_[i][-1].view(-1,1)),0))
                y_real.append(torch.cat((batch_x[i][0].view(-1,1),y_[i]),0))

            y_predict.append(batch_y.data.numpy().flatten().flatten())
            for i in range(len(y_fake)):
                y_fake[i] = y_fake[i].data.numpy()
                y_real[i] = y_real[i].data.numpy()
            y_fake = np.array(y_fake)
            y_real = np.array(y_real)

            y_fake = torch.from_numpy(y_fake.reshape(-1, look_back + 1))
            y_real = torch.from_numpy(y_real.reshape(-1, look_back + 1))
            prob_artist0 = D(y_real)
            prob_artist1 = D(y_fake)

            D_loss = 0.5 * torch.mean(
                torch.mul(prob_artist0 - 1,prob_artist0 - 1))  + 0.5 * torch.mean(
                torch.mul(prob_artist1,prob_artist1))


            G_loss = 0.5 * criterion(y_, batch_y) +   0.5 * torch.mean(
                torch.mul(torch.sub(prob_artist1 , torch.tensor([1.])),torch.sub(prob_artist1 , torch.tensor([1.]))))

            # prob_artist0 = D()
            opt_D.zero_grad()
            D_loss.backward(retain_graph=True)  # retain_graph 这个参数是为了再次使用计算图纸
            opt_D.step()

            # adjust_learning_rate(opt_D, epoch, opt_D_init)
            opt_G.zero_grad()
            G_loss.backward()
            opt_G.step()
            # adjust_learning_rate(opt_G, epoch, opt_G_init)
            print("batch_size |", cnt, "G_loss |", G_loss.data, "D_loss |", D_loss.data)

        #path = "model/gan-lstm" + str(epoch) +".pt"

        path = "model/gan-lstm" + str(epoch) + ".pt"
        torch.save(model, path)
        path2 = "model/D.pt"
        torch.save(model, path)
        torch.save(D, path2)

train_gan()



