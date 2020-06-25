import gluonbook as gb
import math
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import matplotlib.pyplot as plt
from mxnet.gluon import loss as gloss
import pandas as pd
import random
from mxnet import autograd,nd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error


map={}


def data_iter_random(train_x,train_y, na, masks, interval,batch_size, num_steps):

    train_x = train_x.asnumpy()
    train_x.flatten()
    trainX = []
    for i in range(len(train_x[0])):
        trainX.append(train_x[0][i])
    #print(trainX)

    trainY = []
    train_y = train_y.asnumpy()
    train_y.flatten()
    for i in range(len(train_y[0])):
        trainY.append(train_y[0][i])
    #print(trainX)

    L = []
    na = na.asnumpy()
    na.flatten()
    for i in range(len(na[0])):
        L.append(na[0][i])

    masks = masks.asnumpy()
    masks.flatten()
    Masks = []
    for i in range(len(masks[0])):
        Masks.append(masks[0][i])

    interval = interval.asnumpy()
    interval.flatten()
    Interval = []
    for i in range(len(interval[0])):
        Interval.append(interval[0][i])

    num_examples = (len(trainX) - 1) // num_steps
    #print("num_examples",num_examples)

    epoch_size = num_examples // batch_size

    example_indices = list(range(num_examples))
    random.shuffle(example_indices)


    def _dataX(pos):
        return trainX[pos : pos + num_steps] #下标
    def _dataY(pos):
        return trainY[pos: pos + num_steps]

    def _dataL(pos):
        return L[pos: pos + num_steps]

    def _dataM(pos):
        return Masks[pos: pos + num_steps]

    def _dataI(pos):
        return Interval[pos: pos + num_steps]


    for i in range(epoch_size):
        i = i * batch_size
        batch_indices = example_indices[i : i + batch_size]
        X = nd.array(
            [_dataY(j * num_steps) for j in batch_indices])
        Y = nd.array(
            [_dataY(j * num_steps + 1) for j in batch_indices])
        Na = nd.array(
            [_dataL(j * num_steps  ) for j in batch_indices])
        Id = nd.array(
            [_dataX(j * num_steps ) for j in batch_indices])

        M = nd.array(
            [_dataM(j * num_steps) for j in batch_indices])
        I = nd.array(
            [_dataI(j * num_steps) for j in batch_indices])

        yield X, Y,Na,Id,M,I


def data_iter_consecutive(train_x,train_y, na, masks, interval,batch_size, num_steps):

    train_x = train_x.asnumpy()
    train_x.flatten()
    trainX = []
    for i in range(len(train_x[0])):
        trainX.append(train_x[0][i])
    # print(trainX)

    trainY = []
    train_y = train_y.asnumpy()
    train_y.flatten()
    for i in range(len(train_y[0])):
        trainY.append(train_y[0][i])
    # print(trainX)

    L = []
    na = na.asnumpy()
    na.flatten()
    for i in range(len(na[0])):
        L.append(na[0][i])

    masks = masks.asnumpy()
    masks.flatten()
    Masks = []
    for i in range(len(masks[0])):
        Masks.append(masks[0][i])

    interval = interval.asnumpy()
    interval.flatten()
    Interval = []
    for i in range(len(interval[0])):
        Interval.append(interval[0][i])
    data_len = len(trainY)
    batch_len = data_len // batch_size

    trainX,trainY,L,Masks,Interval = nd.array(trainX),nd.array(trainY),nd.array(L),nd.array(Masks),nd.array(Interval)

    dataX = trainX[0: batch_size*batch_len].reshape(( batch_size, batch_len))
    dataY = trainY[0: batch_size*batch_len].reshape(( batch_size, batch_len))
    dataL = L[0: batch_size*batch_len].reshape(( batch_size, batch_len))
    dataM = Masks[0: batch_size*batch_len].reshape(( batch_size, batch_len))
    dataI = Interval[0: batch_size*batch_len].reshape(( batch_size, batch_len))
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = dataY[:, i: i + num_steps]
        Y = dataY[:, i + 1: i + num_steps + 1]

        Na = dataL[:, i : i + num_steps + 1]

        Id = dataX[:, i : i + num_steps + 1]

        M = dataM[:, i: i + num_steps]
        I = dataI[:, i: i + num_steps]

        yield X, Y, Na, Id, M, I


def to_onehot(X):
    #print("ize",size)
    return [x.reshape(-1,1) for x in X.T]
#-----------------------suiji选择 ----------------#

'''
def data_iter(batch_size,train_x,train_y):
    num_examples = len(train_x)
    print("num_examples",num_examples)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        #print("i",i)
        j = nd.array(indices[i:min(i+batch_size,num_examples)])
        yield train_x.take(j),train_y.take(j)
'''
#-----------------------数据格式----------------#


def adam(params, V, S, lr, batch_size):
    """Mini-batch stochastic gradient descent."""

    for i in range(len(params)):
        V[i][:] = 0.9 * V[i][:] + (1-0.9) * params[i].grad
        S[i][:] = 0.999 * S[i][:] + (1-0.999) * params[i].grad * params[i].grad
        V_error = V[i][:] / (1 - 0.9)
        S_error = S[i][:] / (1 - 0.999)

        params[i][:] = params[i] - lr * V_error/(nd.sqrt(S_error)+0.00000001)

#----------------预处理-------------#
def data_preprocessing():

    #df = pd.read_csv("p2.csv", sep=',',nrows=20000)
    df = pd.read_csv("six_na.csv", sep=',')
    #df = pd.read_csv("ES.csv", sep=',')
    df['mask'] = 0
    df.loc[df.hourly_traffic_count > 0, 'mask'] = 1




    Flow = df['hourly_traffic_count'].values
    for i in range(len(Flow)):
        map[i] = Flow[i]


    m = df['mask'].values
    #print("m,", type(m))
    I = m * 0

    for t in range(len(m)):

        if t == 0:
            I[t] = 0
            # tmp = l[t]
        else:
            if m[t] == 0:
                I[t] = 1 + I[t - 1]
            else:
                I[t] = 1

    df['interval'] = I.tolist()


    raw = DataFrame()
    raw['id'] = [x for x in range(len(df['hourly_traffic_count']))]
    raw['flow'] = df['hourly_traffic_count']
    raw['loss'] = df['loss']
    raw['mask'] = df['mask']
    raw['interval'] = df['interval']
    #print(raw)
    values = raw.values
    values = values.astype('float32')

    #print(values)
    return values
    #print(data)


#------------------------梯度裁剪----------------#
def grad_clipping(params,theta):
    norm = nd.array([0])
    for param in params:
        norm += (param.grad ** 2).sum()
    norm = norm.sqrt().asscalar()
    if norm  >  theta:
        for param in params:
            param.grad[:] *= theta / norm

#-----------dense层-----------#
def linreg(X,w,b):
    a = nd.dot(X,w) + b
    #print("预测值y",type(a),a.shape)
    return nd.dot(X,w) + b


#--------------自定义损失函数---------------
def squared_loss(y_hat,y):




    #y_hat = nd.argmax(y_hat,axis=1)
    #y_predict = linreg(y_hat,w,b)


    #print("预测值",y_hat,"真实值" ,y)

    return nd.sum(nd.abs(y_hat- y.reshape(y_hat.shape))) / (len(y_hat))

#------------------------优化算法-------------#
def sgd(params, lr, batch_size):
    """Mini-batch stochastic gradient descent."""
    for param in params:
        print("梯度",param.grad)
        param[:] = param - lr * param.grad / batch_size

def sgd2(params, lr, batch_size):
    """Mini-batch stochastic gradient descent."""
    for i in range(len(params)):
        #print("梯度",param.grad)
        params[i][:] = params[i] - lr * params[i].grad / batch_size

def mnt(params, V,lr, batch_size):
    """Mini-batch stochastic gradient descent."""

    for i in range(len(params)):

        V[i][:] = 0.95 * V[i][:] + lr * params[i].grad / batch_size

        params[i][:] = params[i] - V[i][:]



def adagrad(params, V,lr, batch_size):
    """Mini-batch stochastic gradient descent."""

    for i in range(len(params)):

        V[i][:] = V[i][:] +  params[i].grad * params[i].grad/(batch_size * batch_size)

        params[i][:] = params[i] - (lr *  params[i].grad/(nd.sqrt(V[i][:])+0.000001))



def init_lstm_state(batch_size,num_hiddens):
    return (nd.zeros(shape=(batch_size, num_hiddens)),nd.zeros(shape=(batch_size, num_hiddens)),
            nd.zeros(shape=(batch_size, num_hiddens)),nd.zeros(shape=(batch_size, num_hiddens)))


num_inputs,num_hiddens,num_outputs = 1,100,1
def get_params(batch_size,num_step):
    def _one(shape):
        return nd.random.normal(scale=0.01, shape=shape)

    def _three():
        return (_one((num_inputs, num_hiddens)),
                _one((num_hiddens, num_hiddens)),
                nd.zeros(num_hiddens))



    #----第一层
    W_xi, W_hi, b_i = _three()  # 输入门参数
    W_xf, W_hf, b_f = _three()  # 遗忘门参数
    W_xo, W_ho, b_o = _three()  # 输出门参数
    W_xc, W_hc, b_c = _three()  # 候选记忆细胞参数





    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = nd.zeros(num_outputs)



    w = nd.random.normal(scale=0.01, shape=(1, 1))
    # print("权重",w)
    b = nd.zeros(shape=(1,))  # 可能有错

    # rt参数
    w_r = _one((batch_size, batch_size))
    b_r = nd.zeros(1)

    # 附上梯度
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f,  W_xo, W_ho, b_o,  W_xc, W_hc,
     b_c, W_hq, b_q, w, b, w_r, b_r]

    V = []
    S = []
    for p in params:
        V.append(nd.zeros_like(p))
        S.append(nd.zeros_like(p))
    for param in params:
        param.attach_grad()
    return params, V, S



def lstm(inputs,Masks, Interval,index, state,params,batch_size = 32):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
     b_c, W_hq, b_q, w, b, W_r, b_r] = params
    #print("权重", w)
    (H, H1, C, C1) = state
    outputs = []
  #print("index",index)
    for cnt in range(len(inputs)):
        # print("类型X", X.shape, type(X))
        X = inputs[cnt]
        m = Masks[cnt]
        # print("makk",X.shape)
        r = nd.exp(-nd.maximum(0, nd.dot(W_r, Interval[cnt]) + b_r))
        #print("r",W_xi)
        for i in range(batch_size):

            if m[i][0].asscalar() == 0:
                index_ = index[cnt][i][0].asscalar() - 1
                X[i][0] = (r[i][0] * map[index_] + (1 - r[i][0]) * map[index_ - 299])
                #X[i][0] = (r[i][0] * map[index_] + (1 - r[i][0]) * map[index_ - 1] + (1 - r[i][0]) *(1 - r[i][0])* map[index_ - 47])
                # print("aclove_", index[cnt][i][0].asscalar(), X[i][0].asscalar())
                # if index[cnt][i][0].asscalar() == 12400:
                #    print("aclove",X[i][0])
                # print("aclove", before.asscalar(),X[i][0].asscalar(), m[i][0].asscalar(), index[cnt][i][0].asscalar())
                map[index[cnt][i][0].asscalar()] = X[i][0].asscalar()
                #print("aclove_after",X[i][0])

        inputs[cnt] = X
         #   print('X', r.shape)
        #  X = Masks[cnt] * X + (1 - Masks[cnt]) * r * inputs[cnt - 1] + (1 - Masks[cnt]) * (1 - r) * inputs[cnt - 2]
        #print("r",r)
        #print("makk", H.shape, W_hi.shape)
        I = nd.sigmoid(nd.dot(X, W_xi) + nd.dot(H, W_hi) + b_i)
        F = nd.sigmoid(nd.dot(X, W_xf) + nd.dot(H, W_hf) +  b_f)
        O = nd.sigmoid(nd.dot(X, W_xo) + nd.dot(H, W_ho) +  b_o)
        C_tilda = nd.tanh(nd.dot(X, W_xc) + nd.dot(H, W_hc) +  b_c)
        C = C * r
        C = F * C + I * C_tilda


        H = O * C.tanh()

        Y = nd.dot(H, W_hq) + b_q

        outputs.append(Y)
    return outputs, (H,H, C,C1), w, b


def lstm_train(batch_size,num_step,num_epochs,lr,clipping_theta,flag):
    params, V, S = get_params(batch_size, num_step)



    #loss = squared_loss
    loss = gloss.L2Loss()
    df = data_preprocessing()
    scaler = MinMaxScaler(feature_range=(0, 1))
    features = df[:,0].reshape((1,-1))
    #features = scaler.fit_transform(features)

    features = nd.array(features)

    labels = df[:,1].reshape((1,-1))
    #labels = scaler.fit_transform( labels)
    labels = nd.array(labels)
    que = df[:, 2]
    na = df[:, 2].reshape((1, -1))
    # labels = scaler.fit_transform( labels)
    na = nd.array(na)

    masks = df[:, 3].reshape((1, -1))
    masks = nd.array(masks)

    interval = df[:, 4].reshape((1, -1))
    interval = nd.array(interval)


    #print(features,labels,na)
    loss_value = []


    if flag == 0:#随机采样
        data_iter_fn = data_iter_random
    else:
        data_iter_fn = data_iter_consecutive

    for epoch in range(num_epochs):
        l_sum = 0.0
        n = 0

        if flag == 1:
            state = init_lstm_state(batch_size, num_hiddens)

        data_iter = data_iter_fn(features, labels, na, masks, interval, batch_size, num_step)

        for X, Y ,L, Id, M, I in data_iter:

            if flag == 0:
                state = init_lstm_state(batch_size, num_hiddens)
            else:
                for s in state:
                    s.detach()
            with autograd.record():

                inputs = to_onehot(X)
                Masks = to_onehot(M)
                Interval = to_onehot(I)
                index = to_onehot(Id)
                (outputs, state, w, b) = lstm(inputs, Masks, Interval, index, state, params)
                outputs = nd.concat(*outputs, dim=0)



                y = Y.T.reshape((-1,))
                Lo = L.T.reshape((-1,))
                id = Id.T.reshape((-1,))


                #y = y.relu()
                #--------可去掉

                for i in range(len(y)):
                    if Lo[i][0] == 1:
                        if outputs[i][0] < 0:
                            y[i][0] = map[id[i][0].asscalar()]
                        else:
                            y[i][0] = outputs[i]
                        y[i][0] = max(map[id[i][0].asscalar()], outputs[i][0])
                        map[id[i][0].asscalar()] = y[i][0].asscalar()


                    #map[id[i][0].asscalar()] = y[i][0].asscalar()

                #print("y", y)
                y_predict = outputs
                #y_predict = linreg(outputs, w, b)
                #y_predict = y_predict.relu()

                l = loss(y_predict, y).mean()
                l_sum += l.asscalar()
                n += 1
            l.backward()
            grad_clipping(params, clipping_theta)
            #V_ = V
            #V = mnt(params,V_,lr,batch_size)
            #mnt(params, V, lr, batch_size)
            #adagrad(params, V, lr, batch_size)
            adam(params, V, S, lr, batch_size)
            #print("V",V)
            #sgd(params,lr,batch_size)
            # if (epoch + 1) % pred_period == 0:

        print('epoch %d,mae %lf' % (epoch + 1, l_sum / n))

        loss_value.append(l_sum/n)
        list = []

        # print("map", map, len(map))

        for i in range(len(map)):
            list.append(map[i])

        res = DataFrame()
        res["hourly_traffic_count"] = list
        res["loss"] = que

        #print('---',predict_rnn([10, 11], 10, lstm, params, init_lstm_state, num_hiddens))
        # --------------比较-------------



        #res.to_csv("lstm-m.csv")








    plt.plot(loss_value,label='lr = %f,clip = %f'%(lr,clipping_theta))
    plt.show()




#params = get_params()

#print(predict_rnn([10,20],10,lstm,params,init_lstm_state,num_hiddens))
#lstm_train(32,25,100,0.03,0.001,0) #4 hyper可调  调步长，调学习率，调hidden数量
#lstm_train(32,50,100,0.02,0.005,0)
#lstm_train(32,25,200,0.03,0.001,0)  #0.085 epoch = 17
#lstm_train(32,25,200,0.02,0.001,0)  #0.089 epoch = 22
#lstm_train(32,25,200,0.03,0.001,0)  #0.085 epoch = 22
lstm_train(32,25,200,0.02,0.001,0)
'''
list = []

print("map",map,len(map))

for i in range(len(map)):
    list.append(map[i])

res = DataFrame()
res["hourly_traffic_count"] = list
res.to_csv("LSTM-M.csv")
'''
print("map", map, len(map))

